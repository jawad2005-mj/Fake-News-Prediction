# Fake News Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

[![Open in Streamlit](https://img.shields.io/badge/Open%20in-Streamlit-FF4B4B?style=for-the-badge)](https://fake-news-prediction-4.streamlit.app/)

</div>

This project predicts whether a news item is likely to be real or fake based on the writing style and language patterns in the headline and article text. It includes a Jupyter notebook for the full training workflow and a Streamlit application for interactive use.

Live app: [Fake News Prediction on Streamlit](https://fake-news-prediction-4.streamlit.app/)

## App preview

<p align="center">
  <img src="./assets/app-preview.png" alt="Fake News Detector app preview" width="900">
</p>

## Project overview

The model follows a standard NLP pipeline:

1. Combine the headline and article text.
2. Clean and normalize the text.
3. Remove stopwords and stem words to their root form.
4. Convert text into numerical features using TF-IDF.
5. Train a Logistic Regression classifier.
6. Predict whether the article is likely fake or real.

## Repository structure

```text
Fake_News_Prediction/
├── Project_5_Fake_News_Prediction.ipynb   # end-to-end notebook workflow
├── app.py                                 # Streamlit web app
├── requirements.txt                       # project dependencies
├── README.md                              # project documentation
├── assets/
│   └── app-preview.png                   # UI preview of the fake news detector
├── artifacts/                             # generated model files and cached resources
├── .streamlit/
│   └── config.toml                       # Streamlit UI configuration
└── .venv/                                 # optional local virtual environment (if created)
```

The `artifacts/` directory is created automatically on the first run and stores the trained model so the app loads faster on subsequent launches.

## Dataset

- Source: [rajatkumar30/fake-news](https://www.kaggle.com/datasets/rajatkumar30/fake-news) on Kaggle
- File used: `news.csv`
- Total labeled articles: 6,335
- Distribution: 3,171 real and 3,164 fake
- Columns: `title`, `text`, and `label` (`FAKE` or `REAL`)

The dataset is downloaded automatically in the app using `kagglehub` on first run, so no manual download is required.

## Model details

The training pipeline used in the project is:

1. Fill missing values with empty strings.
2. Convert labels to numeric values: `FAKE = 1`, `REAL = 0`.
3. Merge `title` and `text` into a combined `content` field.
4. Remove non-letter characters, lowercase the text, drop English stopwords, and stem word forms using the Porter stemmer.
5. Transform words into TF-IDF features.
6. Split the data into 80% training and 20% testing with `stratify` and `random_state=2`.
7. Train a Logistic Regression model.

## Model performance

| Dataset | Accuracy |
| --- | --- |
| Training set | 94.9% |
| Test set (unseen articles) | 91.2% |

These values indicate strong performance without major overfitting.

## How to run the project

Requirements:

- Python 3.9 or newer
- Internet access for the first run so the dataset can be downloaded

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2A. Run the notebook

Open `Project_5_Fake_News_Prediction.ipynb` in Jupyter or VS Code and run all cells. The final section allows you to paste your own article text and check the prediction.

### 2B. Run the web app

```bash
streamlit run app.py
```

The application will open in the browser at `http://localhost:8501`.

Notes:

- On the first run, the app downloads the dataset and trains the model. This usually takes approximately 1 to 2 minutes.
- After training, subsequent launches are much faster.
- Use the "Load a random article" button to view a sample from the dataset and compare the model's output to the true label.
- To access the app from a phone or tablet, connect to the same local network and use the Streamlit network URL shown in the terminal.

## Troubleshooting

- If you see a Kaggle download or authentication error, check your internet connection and try again. The dataset is public.
- If the app appears to keep using an outdated model, remove the `artifacts` folder and rerun the application.
- If you encounter `ModuleNotFoundError`, reinstall dependencies with `pip install -r requirements.txt`.

## Limitations

- The system is designed for English-language content.
- It evaluates writing style rather than fact-checking. A convincingly written false story may appear reliable.
- The model was trained on a specific dataset, primarily covering political news, so it may be less reliable on other topics or newer events.

## Technologies used

- Python
- pandas
- NumPy
- scikit-learn
- NLTK
- kagglehub
- Streamlit

## Author

Muhammad Jawad
