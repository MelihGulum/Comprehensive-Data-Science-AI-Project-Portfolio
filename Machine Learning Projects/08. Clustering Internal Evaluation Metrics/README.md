# Customer Personality Analysis – Clustering Comparison Study

## 📌 Project Overview

This project compares multiple clustering algorithms to determine the most suitable method for customer segmentation using internal evaluation metrics.

The dataset used in this study is the **“Customer Personality Analysis”** dataset available on Kaggle.

The objective is to evaluate how different clustering paradigms perform under different data representations and identify the most reliable model.

---

## 🔬 Methodology

### Data Preparation (Summary)

Basic preprocessing steps were applied before clustering:

- Removal of irrelevant and duplicate records  
- Missing value imputation  
- Outlier handling  
- Encoding of categorical variables  
- Feature scaling  
- Feature engineering to enrich customer behavioral representation  

---

### Clusterability Analysis

Before applying clustering algorithms, the dataset’s clustering tendency was evaluated using:

- Principal Component Analysis (PCA)  
- t-SNE visualization  
- Hopkins Statistic  

#### Hopkins Statistic Results

| Dataset        | Hopkins Value |
|---------------|--------------|
| Raw Data       | 0.8593       |
| Processed Data | 0.7983       |

Both values indicate a **strong clustering tendency** (> 0.5), confirming suitability for unsupervised learning.

PCA improved geometric structure and cluster separability, especially for hierarchical models.

---

## 🤖 Clustering Algorithms

Four different clustering paradigms were evaluated:

- **Centroid-based:** K-Means  
- **Density-based:** DBSCAN  
- **Hierarchical:** Agglomerative Clustering  
- **Probabilistic:** Gaussian Mixture Model (GMM)  

Hyperparameter tuning was performed for each algorithm.  
The configuration with the highest Silhouette score was selected as the best-performing model for each dataset version.

---

## 📊 Clustering Performance Comparison

| Dataset        | Model            | #Clusters | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---------------|-----------------|-----------|------------|-------------------|----------------|
| Raw Data      | Agglomerative   | 2         | 0.4637     | 109.9376          | 0.9822        |
| Raw Data      | DBSCAN          | 4         | 0.4074     | 47.8092           | 1.6034        |
| Raw Data      | GMM             | 2         | 0.2250     | 539.9599          | 1.8619        |
| Raw Data      | KMeans          | 2         | 0.2456     | 620.2171          | 1.7333        |
| Processed Data| Agglomerative   | 2         | 0.4474     | 99.1599           | 1.0498        |
| Processed Data| DBSCAN          | 4         | 0.3555     | 39.0988           | 1.9162        |
| Processed Data| GMM             | 2         | 0.3023     | 302.2564          | 1.8179        |
| Processed Data| KMeans          | 2         | 0.2318     | 546.7887          | 1.8572        |
| Raw PCA       | Agglomerative   | 2         | **0.6809** | 16.7194           | **0.2303**    |
| Raw PCA       | DBSCAN          | 3         | 0.4441     | 67.6256           | 2.3130        |
| Raw PCA       | GMM             | 2         | 0.2123     | 495.3525          | 2.0142        |
| Raw PCA       | KMeans          | 2         | 0.2259     | 543.7334          | 1.9171        |
| Processed PCA | Agglomerative   | 2         | 0.4637     | 109.9376          | 0.9822        |
| Processed PCA | DBSCAN          | 4         | 0.4074     | 47.8092           | 1.6034        |
| Processed PCA | GMM             | 2         | 0.2250     | 539.9599          | 1.8619        |
| Processed PCA | KMeans          | 2         | 0.2456     | 620.2171          | 1.7333        |

---

## 🥇 Best Configuration

**Agglomerative Clustering on Raw PCA Data**

- Silhouette Score: **0.6809**
- Davies-Bouldin Index: **0.2303**
- Optimal Number of Clusters: 2

This configuration achieved:

- Strong inter-cluster separation  
- High intra-cluster similarity  
- The most stable overall performance across metrics  

---

## 🔎 Key Insights

- Agglomerative Clustering consistently delivered the most balanced and reliable results.
- K-Means and GMM produced lower separation performance.
- DBSCAN showed moderate performance, indicating partial alignment with dataset density structure.
- PCA significantly improved hierarchical clustering performance.

---

## 🧪 Experimental Environment

All experiments were conducted using:

- Python  
- scikit-learn  
- Google Colab  

The environment ensured reproducibility and computational efficiency.

---

## 🎯 Conclusion

Based on internal validation metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin):

> **Agglomerative Clustering is the most suitable model for this dataset.**

Dimensionality reduction via PCA combined with hierarchical clustering provided the clearest and most reliable customer segmentation structure.