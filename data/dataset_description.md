# Spotify Tracks Dataset

Dataset source: Kaggle, `maharshipandya/-spotify-tracks-dataset`

Local file after download:

```text
data/raw/dataset.csv
```

The dataset contains 114,000 Spotify tracks, 21 columns, and 114 music genres. Each genre has 1,000 tracks, which makes it especially useful for genre comparison, clustering, and multi-class classification.

## Relevant Columns

The project will focus mostly on the columns below:

| Column | Role in the project |
| --- | --- |
| `track_id` | Spotify track identifier. |
| `artists` | Artist names, separated by semicolon when multiple artists are present. |
| `album_name` | Album name. |
| `track_name` | Track title. |
| `popularity` | Target/metric for popularity analysis, from 0 to 100. |
| `duration_ms` | Track duration in milliseconds. |
| `explicit` | Whether the track is explicit. |
| `danceability` | How suitable the track is for dancing, from 0 to 1. |
| `energy` | Perceptual intensity/activity, from 0 to 1. |
| `loudness` | Overall loudness in dB. |
| `speechiness` | Presence of spoken words, from 0 to 1. |
| `acousticness` | Confidence that the track is acoustic, from 0 to 1. |
| `instrumentalness` | Probability that the track has no vocal content, from 0 to 1. |
| `liveness` | Probability that the track was performed live, from 0 to 1. |
| `valence` | Musical positiveness or emotional tone, from 0 to 1. |
| `tempo` | Estimated tempo in BPM. |
| `track_genre` | Genre label. Main target for classification and genre-level analysis. |

## Lower Priority Columns

`key`, `mode`, and `time_signature` may be useful in later feature engineering, but the first project version prioritizes the acoustic and behavioral features above because they are easier to interpret in BI, Analytics, and Data Science storytelling.

`Unnamed: 0` is an index-like column and should be dropped during preprocessing.
