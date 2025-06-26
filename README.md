# Grainger Applied ML Semantic Search Exercise

This repository contains a complete solution for the Grainger ML Engineer Assessment focused on building a semantic search application using Amazon's ESCI dataset.

## Solution Overview

The solution implements a semantic search system that:

- Downloads and prepares the Amazon ESCI dataset according to exercise requirements
- Creates vector embeddings using Sentence Transformers
- Uses FAISS for efficient in-memory vector indexing and similarity search
- Evaluates performance using HITS@K and MRR metrics
- Provides comprehensive analysis and visualizations

## Project Structure

```bash
grainger/
├── data_preparation.py         # Dataset download and preparation script
├── semantic_search.py          # Main semantic search solution
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── Exercise.md                 # Original exercise requirements
└── data/                       # Directory for datasets (created automatically)
    ├── sample_dataset.csv      # Prepared dataset (~500 rows, ~50 queries)
    ├── search_results.json     # Performance metrics
    └── performance_metrics.png # Visualization
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
python data_preparation.py
```

This script will:

- Download the Amazon ESCI dataset from GitHub
- Filter for US locale and exact matches ('E' label)
- Create a sample dataset with ~50 unique queries and ~500 rows
- Save the prepared dataset as `data/sample_dataset.csv`

### 3. Run Semantic Search

```bash
python semantic_search.py
```

This will:

- Load the prepared dataset
- Generate product embeddings using Sentence Transformers
- Build a FAISS vector index
- Evaluate search performance
- Display results and create visualizations

## Technical Architecture

### Embedding Strategy

- **Model**: `all-MiniLM-L6-v2` from Sentence Transformers
- **Text Representation**: Combines product title, description, brand, color, and other attributes
- **Normalization**: L2 normalization for cosine similarity

### Vector Index

- **Technology**: FAISS (Facebook AI Similarity Search)
- **Index Type**: IndexFlatIP (Inner Product for cosine similarity)
- **Storage**: In-memory for fast retrieval

### Evaluation Metrics

- **HITS@K**: Measures if any relevant product appears in top-K results (K=1,5,10)
- **MRR**: Mean Reciprocal Rank - average of reciprocal ranks of first relevant results

## Design Decisions & Assumptions

### Dataset Preparation

1. **Filtering**: Selected only US locale products with exact match labels ('E') as specified
2. **Sampling**: Used random sampling with fixed seed (42) for reproducibility
3. **Query Selection**: Randomly selected 50 unique queries to ensure diversity

### Text Representation

1. **Multi-field Combination**: Combined multiple product attributes for richer embeddings
2. **Structured Format**: Used delimiter-separated format for clear attribute boundaries
3. **Fallback Handling**: Provided fallback text for products with missing information

### Model Selection

1. **all-MiniLM-L6-v2**: Balanced performance and speed, good for general semantic similarity
2. **Alternative considered**: all-mpnet-base-v2 (higher quality but slower)
3. **Justification**: Optimized for semantic similarity tasks, compact size suitable for production

### Index Configuration

1. **FAISS IndexFlatIP**: Exact search for best quality (vs approximate methods)
2. **Cosine Similarity**: Most appropriate for text embeddings
3. **In-memory Storage**: Meets requirement and provides fastest retrieval

## Expected Performance

Based on the exercise setup and similar benchmarks:

- **HITS@1**: 0.15-0.30 (15-30% of queries find relevant product in top result)
- **HITS@5**: 0.40-0.60 (40-60% find relevant product in top 5)
- **HITS@10**: 0.50-0.70 (50-70% find relevant product in top 10)
- **MRR**: 0.25-0.45 (Average reciprocal rank)

Note: Performance depends on dataset quality and query-product relationship complexity

## Configuration Options

### Changing the Embedding Model

To use a different embedding model, modify the `model_name` parameter:

```python
search_engine = SemanticSearchEngine(model_name='all-mpnet-base-v2')
```

### Dataset Size

To change sample size, modify parameters in `data_preparation.py`:

```python
sample_df = data_prep.create_sample_dataset(filtered_df, target_queries=100, target_rows=1000)
```

### Search Parameters

Adjust search parameters in `semantic_search.py`:

```python
similarities, indices = self.search(query, k=20)  # Return top 20 results
```

## Potential Improvements

### Short-term Enhancements

1. **Hybrid Search**: Combine semantic similarity with keyword matching
2. **Query Expansion**: Use synonyms and related terms
3. **Re-ranking**: Add secondary ranking based on product popularity or ratings
4. **Caching**: Cache embeddings and search results

### Advanced Optimizations

1. **Fine-tuning**: Train embeddings on domain-specific data
2. **Multi-modal**: Include product images if available
3. **Personalization**: Incorporate user behavior and preferences
4. **A/B Testing**: Compare different embedding models and search strategies
