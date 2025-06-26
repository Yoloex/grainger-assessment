"""
Data Preparation Script for Amazon ESCI Dataset
This script downloads and prepares train and test dataset.
"""

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class ESCIDataPreparation:
    """
    Class to handle downloading and preparing the Amazon ESCI dataset
    """

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Amazon ESCI dataset URLs
        self.base_url = "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset"
        self.files_to_download = {
            "shopping_queries_dataset_products.parquet": f"{self.base_url}/shopping_queries_dataset_products.parquet",
            "shopping_queries_dataset_examples.parquet": f"{self.base_url}/shopping_queries_dataset_examples.parquet",
        }

    def download_file(self, url, filename):
        """Download a file with progress bar"""
        filepath = self.data_dir / filename

        if filepath.exists():
            print(f"{filename} already exists, skipping download.")
            return filepath

        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(filepath, "wb") as file, tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = file.write(chunk)
                pbar.update(size)

        return filepath

    def download_dataset(self):
        """Download all required dataset files"""
        downloaded_files = {}

        for filename, url in self.files_to_download.items():
            try:
                filepath = self.download_file(url, filename)
                downloaded_files[filename] = filepath
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {str(e)}")
                raise

        return downloaded_files

    def load_datasets(self):
        """Load the downloaded parquet files"""
        products_path = self.data_dir / "shopping_queries_dataset_products.parquet"
        examples_path = self.data_dir / "shopping_queries_dataset_examples.parquet"

        if not products_path.exists() or not examples_path.exists():
            print("Dataset files not found. Downloading...")
            self.download_dataset()

        print("Loading datasets...")
        products_df = pd.read_parquet(products_path)
        examples_df = pd.read_parquet(examples_path)

        print(f"Products dataset shape: {products_df.shape}")
        print(f"Examples dataset shape: {examples_df.shape}")

        return products_df, examples_df

    def filter_dataset(self, products_df, examples_df):
        """
        Filter dataset according to exercise requirements:
        - 'us' product locale
        - 'E' esci_label (Exact match)
        """
        print("Filtering dataset...")

        # Step 1: Filter for US locale and E label
        filtered_examples = examples_df[
            (examples_df["product_locale"] == "us")
            & (examples_df["esci_label"] == "E")
        ].copy()

        print(
            f"After filtering for US locale and E label: {filtered_examples.shape[0]} rows"
        )

        # Merge with products data
        merged_df = filtered_examples.merge(
            products_df, on="product_id", how="inner"
        )

        print(f"After merging with products: {merged_df.shape[0]} rows")
        print(f"Unique queries in filtered dataset: {merged_df['query'].nunique()}")

        return merged_df

    def create_sample_dataset(self, merged_df, target_queries=50, target_rows=500):
        """
        Create sample dataset with approximately 50 unique queries and 500 rows
        """
        print(
            f"Creating sample dataset with ~{target_queries} queries and ~{target_rows} rows..."
        )

        # Step 2a: Get random sample of unique queries
        unique_queries = merged_df["query"].unique()

        if len(unique_queries) < target_queries:
            print(
                f"Warning: Only {len(unique_queries)} unique queries available, using all of them."
            )
            selected_queries = unique_queries
        else:
            np.random.seed(42)  # For reproducibility
            selected_queries = np.random.choice(
                unique_queries, target_queries, replace=False
            )

        print(f"Selected {len(selected_queries)} unique queries")

        # Step 2b: Filter dataset to contain only selected queries
        query_filtered_df = merged_df[
            merged_df["query"].isin(selected_queries)
        ].copy()
        print(f"Dataset with selected queries: {query_filtered_df.shape[0]} rows")

        # Step 2c: Create sample of target_rows from the filtered dataset
        if query_filtered_df.shape[0] <= target_rows:
            print(
                f"Dataset has {query_filtered_df.shape[0]} rows, using all of them."
            )
            sample_df = query_filtered_df.copy()
        else:
            sample_df = query_filtered_df.sample(
                n=target_rows, random_state=42
            ).copy()

        print(
            f"Final sample dataset: {sample_df.shape[0]} rows, {sample_df['query'].nunique()} unique queries"
        )

        return sample_df

    def save_sample_dataset(self, sample_df, filename="sample_dataset.csv"):
        """Save the sample dataset"""
        filepath = self.data_dir / filename
        sample_df.to_csv(filepath, index=False)
        print(f"Sample dataset saved to: {filepath}")
        return filepath

    def analyze_dataset(self, df):
        """Provide basic analysis of the dataset"""
        print("\n" + "=" * 50)
        print("DATASET ANALYSIS")
        print("=" * 50)

        print(f"Dataset shape: {df.shape}")
        print(f"Unique queries: {df['query'].nunique()}")
        print(f"Unique products: {df['product_id'].nunique()}")

        # Product columns analysis
        product_columns = [col for col in df.columns if col.startswith("product_")]
        print(f"\nProduct columns ({len(product_columns)}):")
        for col in product_columns[:10]:  # Show first 10
            print(f"  - {col}")
        if len(product_columns) > 10:
            print(f"  ... and {len(product_columns) - 10} more")

        # Query length analysis
        df["query_length"] = df["query"].str.len()
        print(f"\nQuery length statistics:")
        print(f"  Mean: {df['query_length'].mean():.1f}")
        print(f"  Median: {df['query_length'].median():.1f}")
        print(f"  Min: {df['query_length'].min()}")
        print(f"  Max: {df['query_length'].max()}")  # Top queries
        print(f"\nTop 10 most frequent queries:")
        top_queries = df["query"].value_counts().head(10)
        for query, count in top_queries.items():
            print(f"  '{query}': {count} products")

        print("\n" + "=" * 50)

        return df

    def prepare_full_dataset(self):
        """Complete pipeline to prepare the dataset"""
        print("Starting dataset preparation pipeline...")

        products_df, examples_df = self.load_datasets()
        filtered_df = self.filter_dataset(products_df, examples_df)
        sample_df = self.create_sample_dataset(filtered_df)
        sample_df = self.analyze_dataset(sample_df)
        filepath = self.save_sample_dataset(sample_df)

        print(f"\nDataset preparation completed successfully!")
        print(f"Sample dataset saved to: {filepath}")

        return filepath


def main():
    """Main function to run data preparation"""
    print("Amazon ESCI Dataset Preparation")
    print("=" * 40)

    # Initialize data preparation
    data_prep = ESCIDataPreparation()

    try:
        # Prepare the complete dataset
        data_prep.prepare_full_dataset()

        print(f"\nDataset is ready for semantic search modeling!")
        print(
            f"Next step: Run the semantic search solution using the prepared dataset."
        )

    except Exception as e:
        print(f"Error during data preparation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
