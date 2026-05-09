# Scientific Topic Clustering and Trend Analysis (NLP)
Unsupervised clustering of 150k arXiv abstracts to discover emerging scientific topics and support research collaboration decisions. Built with Python, scikit‑lear and pandas.

**Author:** Olga Ruiz
**Course:** Machine Learning – Unsupervised Learning and Feature Engineering  
**Date:** May 2026  

## Overview

This project addresses a real business problem: a company wants to position itself towards research and academic cooperation. The goal is to identify current scientific trends by analyzing a large corpus of scientific papers from arXiv.

Using unsupervised learning techniques, the project:
- Preprocesses raw text abstracts
- Converts text to numerical vectors using TF‑IDF
- Reduces dimensionality with Truncated SVD (PCA) and L2 normalization
- Applies K‑Means clustering to group papers into 12 coherent research topics
- Compares results with Gaussian Mixture Models (GMM)
- Interprets clusters via top representative words and dominant arXiv categories
- Visualizes results in 2D and analyzes temporal trends (2023–2025)

The final output is a quantitative overview of current scientific trends, delivered as a case study report and this Jupyter notebook.

## Dataset

The dataset used is the arXiv Dataset (metadata of ~1.7 million scientific papers) provided by Cornell University and publicly available on [Kaggle](https://doi.org/10.34740/KAGGLE/DSV/7548853)
For this analysis, a random sample of 150,000 papers published between 2023 and 2025 was used to focus on recent trends.

- **File format:** JSON Lines (`arxiv-metadata-oai-snapshot.json`)
- **Local path:** `data/arxiv-metadata-oai-snapshot.json` (not included in the repository; must be downloaded separately)

## Repository Structure

├── trends_project.ipynb 
├── requirements.txt 
├── README.md 
└── data/ # Folder for the dataset (not uploaded to GitHub)
└── arxiv-metadata-oai-snapshot.json


## Requirements

The code runs in Python 3.11 (or later) and requires the following libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `nltk` 

You can install them with:

pip install -r requirements.txt

## How to Run

1. Download the dataset from Kaggle and place the JSON file inside a data/ folder in the project root.

2. Open trends_project.ipynb with Jupyter Notebook or JupyterLab.

3. Run all cells in order.
The entire pipeline (data loading, cleaning, TF‑IDF, SVD, clustering, visualizations) will execute.

Note: The notebook uses a fixed random seed (38) to ensure reproducible results.

## Key Results

12 coherent scientific clusters were identified, ranging from Quantum Computing to Optimization & Numerical Methods.

Temporal analysis (2023–2025) shows that AI‑related topics, especially Large Language Models (LLMs), are growing fast, while some classical fields (ex: Condensed Matter Physics, Mathematics) are losing share.

A 2D PCA projection reveals a clear L‑shape separating physical sciences from computational/AI topics.

GMM performed poorly (silhouette ~0.006) and was discarded, K‑Means with k=12 was chosen as the final model.

## Final Configuration
|Parameter |	Value|
|----------|--------|
|Years |	2023–2025|
|Sample size |	150,000|
|TF‑IDF max_features |	8,000|
|TF‑IDF min_df |	10|
|TF‑IDF max_df |	0.8|
|Normalization |	L2|
|SVD components |	50|
|Clustering algorithm |	K‑Means|
|Number of clusters (k) |	12|


## Visualizations
The notebook generates the following figures:

- Top 10 arXiv categories

- Histogram of abstract lengths

- Elbow method and silhouette score for k=2 to 15

- 2D scatter plot of the 12 clusters (PCA projection)

- Stacked bar chart of cluster evolution over time (2023–2025)

## Limitations & Future Work

Only abstracts (not full‑text) are used: some detail may be lost.

The SVD reduction retains only 8% of the explained variance, a trade‑off to avoid the curse of dimensionality.

The sample is random (150k papers), small sampling fluctuations are possible.

Future improvements: test text embeddings (ex: BERT, SciBERT), include full papers and try hierarchical clustering on subsamples.

## References

arXiv.org submitters. (2024). arXiv Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/7548853


