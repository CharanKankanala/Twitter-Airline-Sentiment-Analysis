# Twitter Airline Sentiment Analysis

This project trains a sentiment classifier for airline-related tweets using a TF-IDF text representation and logistic regression. The goal is to build a reproducible baseline that handles class imbalance, exposes the main error patterns, and provides a simple inference interface through saved model artifacts.

The dataset contains 14,640 labeled tweets across three classes: negative, neutral, and positive. Negative tweets make up most of the dataset, so the training pipeline uses stratified splits, balanced class weights, and macro-averaged metrics to evaluate performance across all classes.

I kept the model simple on purpose because this assignment was more useful as a clean baseline than as a black-box leaderboard exercise. The interesting part was not getting a perfect score; it was seeing where a traditional text model struggles, especially on neutral tweets.

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

Open the Jupyter URL printed in the terminal. The main notebooks are:

- `sentiment.example.ipynb` for training and evaluation
- `sentiment.API.ipynb` for loading saved artifacts and running predictions

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

## Method Notes

The model uses TF-IDF features with unigrams and bigrams, then trains logistic regression with balanced class weights. I used stratified splits because the neutral and positive classes are much smaller than the negative class, and without stratification the validation set can shift enough to make the metrics noisy.

The first version looked better than it really was because accuracy was dominated by negative tweets. Macro precision, macro recall, and per-class recall were more useful for understanding whether the classifier was learning all three labels or just leaning into the majority class.

Most of the cleanup is intentionally basic: remove URLs, remove mentions, normalize obvious noise, then let TF-IDF handle the remaining vocabulary. That keeps the error analysis readable. When the model misses, it is usually possible to inspect the tweet and understand why.

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
- The model does not use message history, airline metadata, or user-level context.

## Course Context

Built as an MSML 610 text-classification project.
