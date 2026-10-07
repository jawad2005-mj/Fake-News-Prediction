"""
Fake News Detector - Streamlit app
----------------------------------
Same pipeline as the notebook (Project_5_Fake_News_Prediction.ipynb):
    title + text  ->  clean & stem  ->  TF-IDF  ->  Logistic Regression

On the first run the app downloads the dataset with kagglehub and trains the
model (about 1-2 minutes). The trained model is then saved in the "artifacts"
folder, so every later start is instant.

Run with:   streamlit run app.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import joblib
import kagglehub
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 1. Page settings
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",          # centered layout reads well on phone, tablet and desktop
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_FILE = ARTIFACT_DIR / "fake_news_model.joblib"
DATASET = "rajatkumar30/fake-news"


# ---------------------------------------------------------------------------
# 2. Text cleaning (exactly the same steps as the notebook)
# ---------------------------------------------------------------------------
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

stemmer = PorterStemmer()


def stemming(content: str) -> str:
    """Keep letters only, lowercase, remove stopwords, reduce words to their root."""
    content = re.sub("[^a-zA-Z]", " ", content)
    words = content.lower().split()
    words = [stemmer.stem(w) for w in words if w not in STOP_WORDS]
    return " ".join(words)


# ---------------------------------------------------------------------------
# 3. Train the model once, then reuse it
# ---------------------------------------------------------------------------
def train_model() -> dict:
    """Download the data, train the model and return everything the app needs."""
    path = kagglehub.dataset_download(DATASET)
    raw = pd.read_csv(os.path.join(path, "news.csv")).fillna("")
    df = raw.copy()

    df["label"] = df["label"].map({"FAKE": 1, "REAL": 0})     # 1 = fake, 0 = real
    df["content"] = (df["title"] + " " + df["text"]).apply(stemming)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["content"].values)
    Y = df["label"].values

    # Same split as the notebook. "rows" remembers which articles went to the test set.
    rows = np.arange(len(df))
    X_train, X_test, Y_train, Y_test, _, test_rows = train_test_split(
        X, Y, rows, test_size=0.2, stratify=Y, random_state=2
    )

    model = LogisticRegression()
    model.fit(X_train, Y_train)

    # A few unseen articles (original, unstemmed text) for the "Load a random article" button
    samples = raw.iloc[test_rows][["title", "text"]].copy()
    samples["label"] = df.iloc[test_rows]["label"].values
    samples = samples.head(60).reset_index(drop=True)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "train_acc": accuracy_score(model.predict(X_train), Y_train),
        "test_acc": accuracy_score(model.predict(X_test), Y_test),
        "n_articles": len(df),
        "samples": samples,
    }


@st.cache_resource(show_spinner="First run: downloading the dataset and training the model (1-2 minutes)...")
def load_resources() -> dict:
    """Load the saved model if it exists, otherwise train it and save it."""
    if MODEL_FILE.exists():
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass                                   # old or broken file -> train again
    resources = train_model()
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(resources, MODEL_FILE)
    return resources


def predict_fake_probability(text: str, resources: dict) -> float | None:
    """Return the probability (0 to 1) that the article is fake."""
    cleaned = stemming(text)
    if not cleaned:
        return None                                # no English words found
    vector = resources["vectorizer"].transform([cleaned])
    return float(resources["model"].predict_proba(vector)[0][1])   # column 1 = fake


# ---------------------------------------------------------------------------
# 4. Look and feel (CSS). Works on phone, tablet, laptop and desktop.
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --ink: #0e1424;  --panel: #151c30;  --line: #263052;
    --paper: #f1ece1; --muted: #9aa3bd; --gold: #d4b26a;
    --real: #5fd0a5;  --fake: #ff7a7a;
}

html, body, [class*="css"], .stApp { font-family: 'DM Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp { background: var(--ink); }
.block-container { max-width: 860px; padding: 2.2rem 1.2rem 4rem; }

/* hide Streamlit's own chrome for a cleaner look */
#MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; height: 0; }

/* ---------- header ---------- */
.hero { border-top: 3px solid var(--gold); padding-top: 1.4rem; margin-bottom: 1.6rem; }
.hero h1 {
    font-family: 'Fraunces', Georgia, serif; font-weight: 700; color: var(--paper);
    font-size: clamp(2rem, 6vw, 3.4rem); line-height: 1.05; margin: 0 0 .7rem; letter-spacing: -0.01em;
}
.hero p { color: var(--muted); font-size: clamp(.98rem, 2.4vw, 1.1rem); max-width: 56ch; margin: 0; line-height: 1.55; }

/* ---------- stat cards ---------- */
.stat { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: .9rem 1.1rem; margin-bottom: .6rem; }
.stat b { display: block; font-family: 'Fraunces', Georgia, serif; font-size: 1.7rem; color: var(--gold); }
.stat span { color: var(--muted); font-size: .88rem; }

/* ---------- inputs ---------- */
.stTextInput input, .stTextArea textarea {
    background: var(--panel) !important; color: var(--paper) !important;
    border: 1px solid var(--line) !important; border-radius: 10px !important; font-size: 1rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus { border-color: var(--gold) !important; box-shadow: 0 0 0 1px var(--gold) !important; }
.stTextInput label, .stTextArea label { color: var(--paper) !important; font-weight: 500; }

/* ---------- buttons ---------- */
.stButton > button {
    width: 100%; min-height: 3rem; border-radius: 10px; font-weight: 600; font-size: 1rem;
    background: transparent; color: var(--paper); border: 1px solid var(--line);
}
.stButton > button:hover { border-color: var(--gold); color: var(--gold); }
.stButton > button:focus-visible { outline: 2px solid var(--gold); outline-offset: 2px; }
.stButton > button[kind="primary"] { background: var(--gold); color: var(--ink); border: none; }
.stButton > button[kind="primary"]:hover { background: #e3c585; color: var(--ink); }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 1.2rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { color: var(--muted); font-weight: 500; padding: .6rem 0; }
.stTabs [aria-selected="true"] { color: var(--gold) !important; }

/* ---------- result ---------- */
.result { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 1.6rem 1.4rem; margin: 1.2rem 0 .6rem; }
.result.real { border-color: var(--real); }
.result.fake { border-color: var(--fake); }
.stamp {
    display: inline-block; padding: .35rem 1rem; border: 3px double currentColor; border-radius: 8px;
    font-family: 'Fraunces', Georgia, serif; font-weight: 700; font-size: clamp(1.3rem, 5vw, 1.9rem);
    transform: rotate(-4deg); animation: press .35s ease-out;
}
.real .stamp { color: var(--real); }
.fake .stamp { color: var(--fake); }
@keyframes press { from { transform: scale(1.5) rotate(-10deg); opacity: 0; } to { transform: scale(1) rotate(-4deg); opacity: 1; } }
@media (prefers-reduced-motion: reduce) { .stamp { animation: none; } }

.result h3 { font-family: 'Fraunces', Georgia, serif; color: var(--paper); font-size: 1.25rem; margin: 1.1rem 0 .3rem; }
.result p { color: var(--muted); margin: 0 0 1rem; line-height: 1.5; }
.bar { height: 10px; background: #0b101e; border-radius: 999px; overflow: hidden; }
.bar > div { height: 100%; border-radius: 999px; }
.real .bar > div { background: var(--real); }
.fake .bar > div { background: var(--fake); }
.split { display: flex; justify-content: space-between; margin-top: .55rem; color: var(--muted); font-size: .9rem; }
.note { background: #0b101e; border-radius: 10px; padding: .7rem .9rem; margin-top: 1rem; color: var(--paper); font-size: .92rem; }

.small { color: var(--muted); font-size: .85rem; line-height: 1.5; margin-top: 1.2rem; }
.how h4 { font-family: 'Fraunces', Georgia, serif; color: var(--paper); margin: 1.4rem 0 .4rem; }
.how, .how li, .how p { color: var(--muted); line-height: 1.65; }
.how b { color: var(--paper); }

/* ---------- small screens ---------- */
@media (max-width: 640px) {
    .block-container { padding: 1.2rem .9rem 3rem; }
    .result { padding: 1.2rem 1rem; }
    .stat b { font-size: 1.45rem; }
}
</style>
"""


def html(markup: str) -> None:
    """Show a block of HTML (leading spaces removed so Markdown does not treat it as code)."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 5. Small helpers for the page
# ---------------------------------------------------------------------------
def stat_card(value: str, label: str) -> None:
    html(f'<div class="stat"><b>{value}</b><span>{label}</span></div>')


def show_result(fake_prob: float, dataset_label: int | None) -> None:
    """Draw the verdict card."""
    is_fake = fake_prob >= 0.5
    confidence = fake_prob if is_fake else 1 - fake_prob
    css_class = "fake" if is_fake else "real"
    stamp = "Looks fake" if is_fake else "Looks reliable"
    sentence = (
        "The wording and style of this article are close to the fake articles the model learned from."
        if is_fake
        else "The wording and style of this article are close to the real articles the model learned from."
    )

    extra = ""
    if dataset_label is not None:
        actual = "fake" if dataset_label == 1 else "real"
        match = "The model agrees." if (dataset_label == 1) == is_fake else "The model got this one wrong."
        extra = f'<div class="note">The dataset marks this article as <b>{actual}</b>. {match}</div>'

    html(f"""
    <div class="result {css_class}">
        <span class="stamp">{stamp}</span>
        <h3>{confidence:.0%} confident</h3>
        <p>{sentence}</p>
        <div class="bar"><div style="width:{confidence * 100:.0f}%"></div></div>
        <div class="split"><span>Real: {1 - fake_prob:.0%}</span><span>Fake: {fake_prob:.0%}</span></div>
        {extra}
    </div>
    """)


# Buttons change the text boxes through "callbacks" (they run before the page redraws)
def load_random_article(samples: pd.DataFrame) -> None:
    row = samples.sample(1).iloc[0]
    st.session_state.headline = row["title"]
    st.session_state.body = row["text"]
    st.session_state.sample_ref = (row["title"], row["text"], int(row["label"]))


def clear_inputs() -> None:
    st.session_state.headline = ""
    st.session_state.body = ""
    st.session_state.sample_ref = None


# ---------------------------------------------------------------------------
# 6. The page
# ---------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)

html("""
<div class="hero">
    <h1>Fake news detector</h1>
    <p>Paste a headline and its article. The model reads both and tells you whether the
    writing looks like reliable reporting or like fabricated news.</p>
</div>
""")

try:
    resources = load_resources()
except Exception as error:
    st.error(
        "The model could not be prepared. Check your internet connection (the dataset is "
        f"downloaded from Kaggle on the first run) and reload the page.\n\nDetails: {error}"
    )
    st.stop()

st.session_state.setdefault("headline", "")
st.session_state.setdefault("body", "")
st.session_state.setdefault("sample_ref", None)

# Three stat cards (they stack on top of each other on phones)
c1, c2, c3 = st.columns(3)
with c1:
    stat_card(f"{resources['test_acc']:.1%}", "Accuracy on unseen articles")
with c2:
    stat_card(f"{resources['train_acc']:.1%}", "Accuracy on training articles")
with c3:
    stat_card(f"{resources['n_articles']:,}", "Articles the model learned from")

check_tab, how_tab = st.tabs(["Check an article", "How it works"])

# ----- Tab 1: check an article -----
with check_tab:
    b1, b2 = st.columns(2)
    b1.button("Load a random article", on_click=load_random_article, args=(resources["samples"],))
    b2.button("Clear", on_click=clear_inputs)

    headline = st.text_input("Headline", key="headline", placeholder="e.g. Senate passes new budget bill")
    body = st.text_area("Article text", key="body", height=230, placeholder="Paste the full article here...")

    if st.button("Check this article", type="primary"):
        full_text = f"{headline} {body}".strip()

        if len(full_text.split()) < 15:
            st.warning("Add a little more text. The model needs at least 15 words to judge.")
        else:
            fake_prob = predict_fake_probability(full_text, resources)
            if fake_prob is None:
                st.warning("No English words were found. This model only understands English text.")
            else:
                # If the text is an untouched dataset sample, also show the real label
                ref = st.session_state.sample_ref
                dataset_label = ref[2] if ref and ref[0] == headline and ref[1] == body else None
                show_result(fake_prob, dataset_label)

    html("""
    <p class="small">This is a learning project. The model judges writing style, not facts, and it
    was trained mostly on US political news. Do not treat its answer as proof.</p>
    """)

# ----- Tab 2: how it works -----
with how_tab:
    html(f"""
    <div class="how">
    <h4>What happens when you press the button</h4>
    <ol>
        <li>The headline and the article are joined into one text.</li>
        <li>Everything except letters is removed, the text is lowercased and common words such as "the" and "is" are dropped.</li>
        <li>Each word is cut down to its root with the Porter stemmer, so "voting" and "voted" both become "vote".</li>
        <li>TF-IDF turns the words into numbers. Words that are rare but meaningful get more weight.</li>
        <li>A Logistic Regression model turns those numbers into a probability that the article is fake.</li>
    </ol>
    <h4>The data</h4>
    <p>The model learned from <b>{resources['n_articles']:,}</b> labelled news articles
    (<a href="https://www.kaggle.com/datasets/rajatkumar30/fake-news" style="color:#d4b26a">rajatkumar30/fake-news</a> on Kaggle),
    split evenly between real and fake. 80% were used for training and 20% were kept back for testing.</p>
    <h4>Limits to keep in mind</h4>
    <ul>
        <li>It only understands English.</li>
        <li>It learned the style of one collection of articles, so other topics or newer stories may fool it.</li>
        <li>It cannot check facts. A well written lie can look reliable to it.</li>
    </ul>
    </div>
    """)

html('<p class="small">Built with Streamlit and scikit-learn by Muhammad Jawad.</p>')
