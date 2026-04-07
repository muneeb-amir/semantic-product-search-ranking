# Semantic Product Search and Ranking System

A deep learning-based semantic search system that understands user queries and retrieves the most relevant products using transformer-based embeddings and a supervised ranking model.

This project moves beyond keyword matching by learning **semantic relationships between queries and products**, enabling more accurate and meaningful search results.

---

## Overview

This system implements a complete semantic product search pipeline:

* Query understanding using modern NLP techniques
* Product representation using combined title and description
* Multiple embedding strategies (TF-IDF, Word2Vec, FastText, SBERT)
* Supervised deep learning ranking model (SBERT + MLP)
* Real-time search via a web interface

---

## Key Features

### Semantic Search

* Understands intent beyond keywords
* Handles paraphrased queries effectively
* Retrieves contextually relevant products

### Multiple Embedding Techniques

* TF-IDF baseline
* Word2Vec
* FastText
* Sentence-BERT (SBERT)

### Deep Learning Ranking Model

* SBERT embeddings as input
* Feature engineering using:

  * Query embedding (q)
  * Product embedding (p)
  * Absolute difference |q - p|
  * Element-wise product (q ⊙ p)
* Multi-layer perceptron (MLP) for relevance scoring

### Evaluation Metrics

* Precision@K
* Recall@K
* F1@K
* Mean Average Precision (MAP)
* NDCG@K

### Deployment

* Real-time semantic search interface using Gradio

---

## System Architecture

```text
Query → Preprocessing → SBERT Encoding
                                ↓
Product Data → Preprocessing → SBERT Encoding
                                ↓
        Feature Engineering (q, p, |q-p|, q⊙p)
                                ↓
                     MLP Ranking Model
                                ↓
                     Ranked Product Results
```

---

## Dataset

* Amazon ESCI Dataset (query-product pairs)
* Labels:

  * Exact
  * Substitute
  * Complement
  * Irrelevant

### Preprocessing Steps

* Lowercasing
* Stopword removal
* Lemmatization
* Removal of special characters and noise

Product representation:

```
product_text = product_title + product_description
```

---

## Model Architecture

### Embedding Model

* Sentence-BERT: `all-MiniLM-L6-v2`
* 384-dimensional embeddings

### Ranking Model (MLP)

```
Input: 4d vector
→ Dense (512) + ReLU
→ Dense (128) + ReLU
→ Output (1)
```

* Loss: Mean Squared Error
* Optimizer: Adam

---

## Training Strategy

* Dataset split:

  * 70% training
  * 15% validation
  * 15% test

* Split performed by **query_id** to avoid leakage

* Hyperparameter tuning:

  * Learning rate: {1e-3, 2e-4}
  * Batch size: {64, 128}

---

## Results

The SBERT + MLP model outperforms all baseline methods:

* Higher Precision@K and Recall@K
* Improved MAP and NDCG scores
* Better semantic alignment compared to TF-IDF and Word2Vec

The training curve (see report) shows stable convergence with decreasing validation loss.

---

## Web Application

Built using Gradio:

### Features

* Text query input
* Real-time semantic search
* Ranked product results
* Adjustable top-K results

### Example Queries

* "wireless earbuds"
* "gaming keyboard rgb"
* "kids educational tablet"

The interface demonstrates strong semantic matching between queries and products.

---

## Project Structure

```text
.
## Project Structure

```
semantic-product-search-ranking/
│
├── app.py                  # Gradio web application for real-time semantic search
├── semantic_search.ipynb   # Complete training pipeline (preprocessing, embeddings, model training, evaluation)
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
│
└── sample_data/ (optional) # Small subset of dataset for demonstration (if included)
```

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/your-username/semantic-product-search-ranking.git
cd semantic-product-search-ranking
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
python app.py
```

---

## Usage

1. Enter a product query
2. System encodes query using SBERT
3. Computes relevance scores via MLP
4. Returns ranked products

---

## Key Highlights

* End-to-end deep learning pipeline
* Strong understanding of semantic search systems
* Integration of transformer embeddings with supervised ranking
* Real-world dataset (Amazon ESCI)
* Deployment-ready web application

---
