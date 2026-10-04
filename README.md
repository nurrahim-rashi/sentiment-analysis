# Sentiment Analysis of Gojek App Reviews (Google Play Store)

A three-class sentiment classifier (negative, neutral, positive) for Indonesian-language app reviews, built end to end:
scraping, preprocessing, lexicon-based labeling, three training schemes, error analysis, hyperparameter tuning, and model interpretation.

**Author:** Rashifa Ashri Nurrahim

## Results

| Model | Features | Split | Train accuracy | Test accuracy | Test macro F1 |
|---|---|---|---|---|---|
| BiLSTM | Word embedding | 80/20 | 97.85% | 94.12% | 0.895 |
| SVM (linear) | TF-IDF | 80/20 | 97.51% | 93.20% | 0.857 |
| Logistic Regression | Bag of Words | 70/30 | 99.42% | 94.23% | 0.897 |
| SVM (linear), tuned with grid search | TF-IDF | 80/20 | n/a | 94.37% | 0.894 |

Numbers come from the executed notebook in this repository. BiLSTM results vary slightly between runs and hardware.

## Dataset

- 100,000 public reviews of the Gojek app, scraped from Google Play Store with `google-play-scraper` (22 Oct 2024 to 2 Oct 2026)
- 60,497 reviews remain after removing duplicates and empty texts
- Labels are assigned with the InSet Indonesian sentiment lexicon: 37,673 negative, 16,630 positive, 6,194 neutral
- User names and profile pictures are not stored

## Key findings

1. **Errors concentrate on borderline reviews.** Reviews whose lexicon score is close to zero are misclassified about 12 times more often
   than the rest, and about 80% of all BiLSTM errors involve the neutral class.
2. **Label quality is the main limitation.** About 37% of 4-5 star reviews receive a negative lexicon label. The reported accuracy measures
   agreement with the lexicon, not with human judgment.
3. **Common preprocessing advice did not help here.** Stemming and negation merging lowered SVM accuracy by 2 to 4 points, because the labels
   were derived from single, unstemmed words.
4. **Augmenting the minority class worked.** Label-checked synonym replacement, random insertion, and random swap raised neutral recall
   from 0.55 to 0.65.
5. **Tuning closed the gap to deep learning.** Grid search (`C`, n-gram range, class weights) raised the SVM from 93.2% to 94.4% test accuracy
   and neutral recall from 0.55 to 0.74, matching the BiLSTM at a fraction of the training time.

## Repository structure

| File | Purpose |
|---|---|
| `scraping_playstore.ipynb` | Collects the reviews and writes `dataset_ulasan.csv` |
| `analisis_sentimen.ipynb` | Preprocessing, labeling, training, evaluation, error analysis, tuning, interpretation |
| `dataset_ulasan.csv` | The scraped dataset |
| `requirements.txt` | Library versions used for the executed notebook |

Notebook sections: 1-6 data preparation, 7-9 the three training schemes, 10 comparison, 11 inference, 12 error analysis,
13 advanced preprocessing and augmentation, 14 hyperparameter optimization, 15 interpretability (feature weights, LIME, word clouds, t-SNE).

The reviews are in Indonesian, so class names (`negatif`, `netral`, `positif`), variable names, and printed output from the original
pipeline are in Indonesian. All explanations are in English.

## Installation

Python 3.10 or newer.

```bash
git clone <repository-url>
cd <repository-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
```

## Running the pipeline

1. **(Optional) Re-scrape the data.** Run `scraping_playstore.ipynb`. This overwrites `dataset_ulasan.csv` with newer reviews, so the results
   will differ from the ones above. Skip this step to reproduce the reported numbers.
2. **Run the analysis.** Open `analisis_sentimen.ipynb` and run all cells from top to bottom.

```bash
jupyter notebook analisis_sentimen.ipynb
```

The notebook also runs on Google Colab: upload both the notebook and `dataset_ulasan.csv`, then choose *Runtime > Run all*.
A full run takes roughly 10 to 15 minutes on a laptop CPU; the BiLSTM trains faster with a GPU.
