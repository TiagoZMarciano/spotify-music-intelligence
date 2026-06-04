# Spotify Music Intelligence

As someone with a strong interest in music, I became particularly interested in understanding how musical structure can be quantified when I discovered that Spotify provides audio feature data. This project explores how audio features can be used to model musical structure, predict genre, and build a recommendation engine.

## Objective

Build an end-to-end analytics and data science portfolio project that demonstrates BI thinking, exploratory analysis, supervised modeling, and applied recommendation systems using Spotify audio features.

## Dataset

The project uses the Kaggle dataset `maharshipandya/-spotify-tracks-dataset`.

To download it locally:

```python
import kagglehub

path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
print("Path to dataset files:", path)
```

## Project Roadmap

### Phase 1: Exploratory Audio Intelligence

Portfolio focus: BI, analytics storytelling, and exploratory data science.

Key questions:

- What differentiates music genres?
- Which genres are more energetic, danceable, acoustic, or emotionally positive?
- Is there a relationship between popularity and valence?
- Are explicit tracks more popular?
- Do genres naturally cluster based on audio features?

Techniques:

- EDA
- Correlation analysis
- PCA
- KMeans clustering
- 2D visualization with dimensionality reduction

### Phase 2: Genre Classification

Problem: Given only acoustic attributes, can we predict a track's genre?

Models to compare:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

Evaluation:

- Accuracy
- Macro and weighted F1-score
- Confusion matrix
- Feature importance

### Phase 3: Recommendation System

Recommendation approaches:

- Content-based filtering using audio feature similarity
- Mood-based recommendation using interpretable audio rules

Example:

```text
User wants happy and energetic music
-> high valence
-> high energy
-> medium/high tempo
```

## Music Behavior & Emotional Modeling

This project also connects audio features to behavioral and emotional interpretation:

- `valence`: emotional positivity
- `energy`: intensity and activation
- `danceability`: engagement and movement potential
- `tempo`: perceived speed and excitement
- `acousticness`: organic/acoustic character
- `speechiness`: spoken-word presence

The goal is not to reduce music to a single number, but to show how quantitative features can support meaningful musical and behavioral analysis.

## Repository Structure

I structured this repository thinking on a cleaner experience of visualization, so I let most of functions separeted in the 'src' folder with its explanations

```text
spotify-music-intelligence/
├── README.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_clustering.ipynb
│   ├── 03_genre_classification.ipynb
│   └── 04_recommendation_system.ipynb
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   └── recommender.py
├── data/
│   ├── dataset_description.md
│   └── raw/
│       └── dataset.csv
├── reports/
│   └── figures/
└── requirements.txt
```

## Getting Started

```bash
pip install -r requirements.txt
```

Then download the dataset with KaggleHub and place/copy `dataset.csv` into `data/raw/`.


