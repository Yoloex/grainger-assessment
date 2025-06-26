"""
Semantic Search Solution
This script implements a semantic search system using FAISS for vector indexing
and sentence transformers for embedding generation.
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from tqdm import tqdm
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings("ignore")


class SemanticSearchEngine:
    """
    Semantic Search Engine using FAISS for vector indexing and Sentence Transformers for embeddings
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", data_dir: str = "data"):
        """
        Initialize the semantic search engine

        Args:
            model_name: Name of the sentence transformer model to use
            data_dir: Directory containing the dataset
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.encoder = None
        self.faiss_index = None
        self.product_df = None
        self.query_df = None
        self.product_embeddings = None

        print(f"Initializing Semantic Search Engine with model: {model_name}")

    def load_encoder(self):
        """Load the sentence transformer model"""
        print("Loading sentence transformer model...")
        self.encoder = SentenceTransformer(self.model_name)
        print(f"Model loaded: {self.model_name}")

    def load_dataset(self, dataset_path: str = None):
        """Load the prepared dataset"""
        if dataset_path is None:
            dataset_path = self.data_dir / "sample_dataset.csv"

        print(f"Loading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)

        # Separate unique products and queries for analysis
        self.product_df = df.drop_duplicates("product_id").copy()
        self.query_df = df[["query", "product_id"]].copy()

        print(f"Dataset loaded: {df.shape[0]} rows")
        print(f"Unique products: {len(self.product_df)}")
        print(f"Unique queries: {df['query'].nunique()}")

        return df

    def create_product_text(self, row):
        """
        Create searchable text from product attributes
        This is a key design decision - combining relevant product fields
        """
        text_fields = []

        if "product_title" in row and pd.notna(row["product_title"]):
            text_fields.append(str(row["product_title"]))

        if "product_description" in row and pd.notna(row["product_description"]):
            text_fields.append(str(row["product_description"]))

        if "product_brand" in row and pd.notna(row["product_brand"]):
            text_fields.append(f"Brand: {str(row['product_brand'])}")

        if "product_color" in row and pd.notna(row["product_color"]):
            text_fields.append(f"Color: {str(row['product_color'])}")

        product_attrs = [
            col
            for col in row.index
            if col.startswith("product_")
            and col
            not in [
                "product_id",
                "product_title",
                "product_description",
                "product_brand",
                "product_color",
                "product_locale",
            ]
        ]

        for attr in product_attrs:
            if pd.notna(row[attr]) and str(row[attr]).strip():
                attr_name = attr.replace("product_", "").replace("_", " ").title()
                text_fields.append(f"{attr_name}: {str(row[attr])}")

        product_text = " | ".join(text_fields)
        return product_text if product_text else "No product information available"

    def generate_embeddings(self, df):
        """Generate embeddings for products"""
        print("Generating product embeddings...")

        if self.encoder is None:
            self.load_encoder()

        print("Creating product text representations...")
        product_texts = []
        for _, row in tqdm(self.product_df.iterrows(), total=len(self.product_df)):
            product_text = self.create_product_text(row)
            product_texts.append(product_text)

        print("Encoding product texts...")
        self.product_embeddings = self.encoder.encode(
            product_texts, show_progress_bar=True, batch_size=32
        )

        print(f"Generated embeddings: {self.product_embeddings.shape}")
        return self.product_embeddings

    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        print("Building FAISS index...")

        if self.product_embeddings is None:
            raise ValueError(
                "Product embeddings not generated. Call generate_embeddings first."
            )

        embedding_dim = self.product_embeddings.shape[1]

        self.faiss_index = faiss.IndexFlatIP(embedding_dim)

        faiss.normalize_L2(self.product_embeddings)

        self.faiss_index.add(self.product_embeddings.astype("float32"))

        print(f"FAISS index built with {self.faiss_index.ntotal} vectors")

    def search(self, query: str, k: int = 10) -> Tuple[List[float], List[int]]:
        """
        Search for similar products given a query

        Args:
            query: Search query string
            k: Number of results to return

        Returns:
            Tuple of (similarities, product_indices)
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_faiss_index first.")

        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)

        similarities, indices = self.faiss_index.search(
            query_embedding.astype("float32"), k
        )

        return similarities[0].tolist(), indices[0].tolist()

    def evaluate_search_performance(self, df, k_values: List[int] = [1, 5, 10]):
        """
        Evaluate search performance using HITS@K and MRR metrics

        Args:
            df: Dataset with query-product pairs
            k_values: List of k values for HITS@K evaluation
        """
        print("Evaluating search performance...")

        query_groups = df.groupby("query")["product_id"].apply(list).to_dict()

        results = {f"HITS@{k}": [] for k in k_values}
        results["MRR"] = []

        all_queries = list(query_groups.keys())

        for query in tqdm(all_queries, desc="Evaluating queries"):
            relevant_product_ids = set(query_groups[query])

            try:
                _, indices = self.search(query, k=max(k_values))

                returned_product_ids = [
                    self.product_df.iloc[idx]["product_id"] for idx in indices
                ]

                # Calculate HITS@K for different k values
                for k in k_values:
                    top_k_products = set(returned_product_ids[:k])
                    hits = len(top_k_products.intersection(relevant_product_ids))
                    results[f"HITS@{k}"].append(1 if hits > 0 else 0)

                # Calculate MRR (Mean Reciprocal Rank)
                reciprocal_rank = 0
                for i, product_id in enumerate(returned_product_ids):
                    if product_id in relevant_product_ids:
                        reciprocal_rank = 1.0 / (i + 1)
                        break
                results["MRR"].append(reciprocal_rank)

            except Exception as e:
                print(f"Error processing query '{query}': {str(e)}")
                # Add zeros for failed queries
                for k in k_values:
                    results[f"HITS@{k}"].append(0)
                results["MRR"].append(0)

        avg_results = {}
        for metric, values in results.items():
            avg_results[metric] = np.mean(values) if values else 0

        return avg_results, results

    def display_search_examples(self, df, num_examples: int = 5):
        """Display some search examples with results"""
        print(f"\n{'='*60}")
        print("SEARCH EXAMPLES")
        print(f"{'='*60}")

        unique_queries = df["query"].unique()
        sample_queries = np.random.choice(
            unique_queries, min(num_examples, len(unique_queries)), replace=False
        )

        for i, query in enumerate(sample_queries, 1):
            print(f"\nExample {i}: '{query}'")
            print("-" * 40)

            ground_truth = df[df["query"] == query]["product_id"].unique()
            print(f"Ground truth products: {len(ground_truth)}")

            try:
                similarities, indices = self.search(query, k=5)

                print("Top 5 search results:")
                for j, (sim, idx) in enumerate(zip(similarities, indices)):
                    product_row = self.product_df.iloc[idx]
                    product_id = product_row["product_id"]
                    product_title = product_row.get("product_title", "N/A")

                    is_relevant = (
                        "[RELEVANT]"
                        if product_id in ground_truth
                        else "[NOT RELEVANT]"
                    )

                    print(
                        f"  {j+1}. {is_relevant} (Score: {sim:.3f}) {product_title[:60]}..."
                    )

            except Exception as e:
                print(f"Error searching for query: {str(e)}")

    def save_results(self, results: Dict, filename: str = "search_results.json"):
        """Save evaluation results to file"""
        filepath = self.data_dir / filename
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {filepath}")

    def create_visualizations(self, results: Dict):
        """Create visualizations of the results"""
        print("Creating visualizations...")

        metrics = []
        values = []
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                metrics.append(metric)
                values.append(value)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            metrics, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        )
        plt.title(
            "Semantic Search Performance Metrics", fontsize=16, fontweight="bold"
        )
        plt.ylabel("Score", fontsize=12)
        plt.xlabel("Metrics", fontsize=12)

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.ylim(0, max(values) * 1.2)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plot_path = self.data_dir / "performance_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Visualization saved to: {plot_path}")


def main():
    print("Grainger Semantic Search Solution")
    print("=" * 50)

    search_engine = SemanticSearchEngine()

    try:
        df = search_engine.load_dataset()

        search_engine.load_encoder()
        search_engine.generate_embeddings(df)
        search_engine.build_faiss_index()

        print("\nEvaluating search performance...")
        avg_results, _ = search_engine.evaluate_search_performance(df)

        print(f"\n{'='*50}")
        print("PERFORMANCE RESULTS")
        print(f"{'='*50}")
        for metric, value in avg_results.items():
            print(f"{metric:10}: {value:.4f}")

        # Show search examples
        search_engine.display_search_examples(df)

        # Save results
        all_results = {
            "average_metrics": avg_results,
            "model_name": search_engine.model_name,
            "dataset_info": {
                "total_rows": len(df),
                "unique_products": len(search_engine.product_df),
                "unique_queries": df["query"].nunique(),
            },
        }
        search_engine.save_results(all_results)

        # Create visualizations
        search_engine.create_visualizations(avg_results)

        print(f"\nSemantic search solution completed successfully!")
        print(f"Performance Summary:")
        print(f"   - HITS@1:  {avg_results['HITS@1']:.4f}")
        print(f"   - HITS@5:  {avg_results['HITS@5']:.4f}")
        print(f"   - HITS@10: {avg_results['HITS@10']:.4f}")
        print(f"   - MRR:     {avg_results['MRR']:.4f}")

    except Exception as e:
        print(f"Error in semantic search pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
