# Music Genre Discovery & Audio Feature Clustering: Unsupervised learning analysis of Spotify track data
## CSCA 5632 Unsupervised Algorithms in Machine Learning final project by Rylan Turner

## Project overview
This project utilizes unsupervised machine learning to discover latent music genres from Spotify audio features and produces an interesting insight: the analysis of audio features suggest **music exists in just 2 fundamental categories**, and crucially, that traditional genre labels do not reflect audio similarity.

## Key discovery
**114 Traditional Genres → 2 Audio-Based Clusters**

- **Cluster 0 (99.9%)**: Mainstream music spanning multiple traditional genres
- **Cluster 1 (0.1%)**: Specialized music (primarily sleep/ambient)
- **Insight**: Traditional genre labels are more cultural/marketing-driven than audio-based

## Results summary
| Algorithm | Clusters | Silhouette Score | Key Strength |
|-----------|----------|------------------|--------------|
| **DBSCAN** | 2 | **0.299** | Best separation + noise detection |
| K-means | 2 | 0.260 | Simple, effective baseline |
| Hierarchical | 2 | 0.239 | Interpretable tree structure |
| GMM | 15 | 0.011 | Probabilistic soft boundaries |

**Winner**: DBSCAN with highest silhouette score and noise detection capability.


## Problem statement
**Challenge**: Traditional music genres create inconsistent categorization for recommendation systems. A song labeled "rock" in one culture might be "pop" in another.

**Question**: Can we discover meaningful music categories using only audio features, without human-imposed genre labels?

**Approach**: Apply multiple unsupervised clustering algorithms to Spotify audio features and compare results.


## Setup instructions
**Note**: This analysis uses the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) from Kaggle. 
For full reproduction, download the dataset and place `spotify_tracks.csv` in the project root.

### Clone repository
`git clone https://www.github.com/rylanturner02/music-genre-clustering.git`
`cd music-genre-clustering`

### Install dependencies
`pip install -r requirements.txt`

### Launch Jupyter
`jupyter notebook music-genre-analysis.ipynb`

## Repository contents
- **music-genre-analysis.ipynb** - Jupyter notebook with analysis and visualizations
- **clustering_results.pkl** - Saved model results and parameters
- **X_scaled.npy**, **X_sample_scaled.npy** - Processed feature matrices
- **X_pca.npt**, **X_tsne.npy** - Dimensionality reduction outputs
- **requirements.txt** - Python dependencies
