# Twitter Airline Sentiment Analysis

This project trains a sentiment classifier for airline-related tweets using a TF-IDF text representation and logistic regression. The goal is to build a reproducible baseline that handles class imbalance, exposes the main error patterns, and provides a simple inference interface through saved model artifacts.

The dataset contains 14,640 labeled tweets across three classes: negative, neutral, and positive. Negative tweets make up most of the dataset, so the training pipeline uses stratified splits, balanced class weights, and macro-averaged metrics to evaluate performance across all classes.

## Project Summary

- Dataset: airline sentiment tweets
- Classes: negative, neutral, positive
- Model: TF-IDF features with logistic regression
- Evaluation: accuracy, macro precision, macro recall, macro F1, confusion matrix, and per-class analysis
- Reproducibility: fixed random seed, saved artifacts, and Docker-based notebook environment
- Deliverables: training notebook, inference notebook, reusable utility module, trained vectorizer, trained model, and EDA report

## Current Result

The test-set baseline reaches approximately:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.776 |
| Macro precision | 0.725 |
| Macro recall | 0.744 |
| Macro F1 | 0.731 |

Per-class behavior:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Negative | 0.83 | 0.82 |
| Neutral | 0.75 | 0.68 |
| Positive | 0.79 | 0.84 |

Neutral tweets are the hardest class because they are less frequent and often sit close to the boundary between mild praise, mild complaint, and factual status updates.

## Quick Start

Build the Docker image:

```bash
bash docker_build.sh
```

Start Jupyter:

```bash
bash docker_jupyter.sh
```

Open the Jupyter URL printed in the terminal, then run:

1. `sentiment.example.ipynb` to train and evaluate the model
2. `sentiment.API.ipynb` to load the saved artifacts and run predictions

## Repository Structure

```text
data/Tweets.csv                  labeled tweet dataset
sentiment_utils.py               preprocessing, training, evaluation, and inference utilities
sentiment.example.ipynb          training and evaluation notebook
sentiment.example.md             exported training notes
sentiment.API.ipynb              inference demonstration notebook
sentiment.API.md                 exported inference notes
tfidf_vectorizer.joblib          saved TF-IDF vectorizer
logreg_sentiment_model.joblib    saved logistic regression model
airline_sentiment_profile.html   YData profiling report
requirements.txt                 Python dependencies
Dockerfile                       container environment
docker_*.sh                      helper scripts
```

## Method

The training pipeline follows a standard supervised text-classification workflow:

1. Load the tweet dataset.
2. Clean text by removing URLs, mentions, and non-informative characters.
3. Encode labels as negative, neutral, and positive.
4. Split data into train, validation, and test sets using stratification.
5. Fit a TF-IDF vectorizer with unigrams and bigrams.
6. Train logistic regression with balanced class weights.
7. Evaluate overall and per-class metrics.
8. Review confusion patterns and influential terms.
9. Save the vectorizer and classifier for reuse.

## Inference Example

```python
api = SentimentAPI(vectorizer, model)

sentiment = api.predict("Great flight and friendly staff.")
probabilities = api.predict_with_confidence("Good flight, but the delay was frustrating.")
```

## Docker Commands

```bash
bash docker_build.sh      # build the image
bash docker_jupyter.sh    # start Jupyter
bash docker_bash.sh       # open a shell in the container
bash docker_clean.sh      # clean local container artifacts
```

## Limitations

- The model is trained only on airline tweets.
- Sarcasm and indirect complaints are difficult for TF-IDF features.
- Neutral sentiment has lower recall than the positive and negative classes.
- Each tweet is classified independently, without conversation context.
- A transformer-based model would likely improve recall, but this project intentionally keeps the baseline simple and interpretable.

## Course Context

Built as an MSML 610 text-classification project.
