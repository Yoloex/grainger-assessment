"""
Main runner script for the Grainger Semantic Search Exercise
This script runs the complete pipeline: data preparation + semantic search
"""

import sys
from pathlib import Path


def run_pipeline():
    """Run the complete semantic search pipeline"""
    print("Starting Grainger Semantic Search Pipeline")
    print("=" * 60)

    try:
        # Step 1: Data Preparation
        print("Step 1: Preparing dataset...")
        print("-" * 30)

        from data_preparation import ESCIDataPreparation

        data_prep = ESCIDataPreparation()
        dataset_path = data_prep.prepare_full_dataset()

        print(f"Dataset prepared successfully!")
        print(f"   - Dataset saved to: {dataset_path}")

        # Step 2: Semantic Search
        print(f"\nStep 2: Running semantic search...")
        print("-" * 30)

        from semantic_search import SemanticSearchEngine

        search_engine = SemanticSearchEngine()

        # Load dataset
        df = search_engine.load_dataset(dataset_path)

        search_engine.load_encoder()
        search_engine.generate_embeddings(df)
        search_engine.build_faiss_index()
        avg_results, _ = search_engine.evaluate_search_performance(df)

        # Display results
        print(f"\nFINAL RESULTS")
        print("=" * 40)
        for metric, value in avg_results.items():
            print(f"{metric:12}: {value:.4f}")

        # Show examples
        search_engine.display_search_examples(df, num_examples=3)

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

        print(f"\nPipeline completed successfully!")
        print(f"Key Metrics:")
        print(
            f"   - HITS@1:  {avg_results['HITS@1']:.4f} ({avg_results['HITS@1']*100:.1f}%)"
        )
        print(
            f"   - HITS@5:  {avg_results['HITS@5']:.4f} ({avg_results['HITS@5']*100:.1f}%)"
        )
        print(
            f"   - HITS@10: {avg_results['HITS@10']:.4f} ({avg_results['HITS@10']*100:.1f}%)"
        )
        print(f"   - MRR:     {avg_results['MRR']:.4f}")

        return True

    except ImportError as e:
        print(f"Missing dependencies. Please install requirements:")
        print(f"   pip install -r requirements.txt")
        print(f"   Error: {str(e)}")
        return False

    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        print(
            f"   Check the error details above and ensure all requirements are met."
        )
        return False


def check_requirements():
    """Check if required files exist"""
    required_files = [
        "requirements.txt",
        "data_preparation.py",
        "semantic_search.py",
    ]
    missing_files = []

    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"Missing required files: {', '.join(missing_files)}")
        return False

    return True


def main():
    print("Grainger Applied ML Semantic Search Exercise")
    print("Building semantic search solution with FAISS\n")

    if not check_requirements():
        return

    response = input(
        "This will download ~100MB of data and may take 5-10 minutes. Continue? (y/n): "
    )
    if response.lower() not in ["y", "yes"]:
        print("Operation cancelled.")
        return

    success = run_pipeline()

    if success:
        print(f"\nAll done! Check the 'data' folder for:")
        print(f"   - sample_dataset.csv (prepared dataset)")
        print(f"   - search_results.json (performance metrics)")
        print(f"   - performance_metrics.png (visualization)")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
