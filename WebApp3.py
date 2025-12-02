import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import base64
import altair as alt
import sys
from pathlib import Path

# ----------------------------
# Config / Paths (relative)
# ----------------------------
MODEL_FILE_ID = "1NQUaA1gcgCCUXlRcpHnh1b3z5dnB5ehz"
VECTOR_FILE_ID = "1clQI6rendrGCOsC9bQGYHg_JRmqiGhWz"

MODEL_PATH = "rf_model_final.pkl"
VECTORIZER_PATH = "vectorizer_final.pkl"

# Images (place these files in the repo root or the same folder as this file)
HOME_LOGO_PATH = "homeplogo.png"
SIDEBAR_LOGO_PATH = "ccitlogo.png"
BACKGROUND_IMAGE_PATH = "bg_final.png"

# Visual constants
LIGHT_BG = "#e8f3f8"
DARK_PRIMARY = "#1f3a52"
ACCENT_BLUE = "#1f7fc1"

# Build Drive "uc" URLs (gdown friendly)
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
VECTOR_URL = f"https://drive.google.com/uc?id={VECTOR_FILE_ID}"

# ----------------------------
# Utility functions
# ----------------------------

def get_base64_of_file(path: str):
    """Return base64 string for an existing file. Returns None if file missing."""
    try:
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None


def download_if_missing(url: str, out_path: str):
    """Download a file using gdown if it's missing.
    Assumes gdown is in requirements.txt and installed in the environment.
    """
    if os.path.exists(out_path):
        return True
    try:
        import gdown
    except Exception as e:
        st.error("Required package 'gdown' is missing. Add it to requirements.txt and redeploy.")
        return False

    try:
        st.info(f"Downloading {Path(out_path).name}...")
        gdown.download(url, out_path, quiet=False)
        return os.path.exists(out_path)
    except Exception as e:
        st.error(f"Failed to download {out_path}: {e}")
        return False


@st.cache_resource
def load_model_and_vectorizer():
    """Download (if needed) and load model + vectorizer with joblib.
    Returns (model, vectorizer) tuple or (None, None) on failure.
    """
    ok_vec = download_if_missing(VECTOR_URL, VECTORIZER_PATH)
    ok_model = download_if_missing(MODEL_URL, MODEL_PATH)

    if not ok_vec or not ok_model:
        return None, None

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        model = None

    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
    except Exception as e:
        st.error(f"Could not load vectorizer: {e}")
        vectorizer = None

    return model, vectorizer


# ----------------------------
# App initialization
# ----------------------------
HISTORY_FILE = "sentiment_history.csv"
if not os.path.exists(HISTORY_FILE):
    try:
        pd.DataFrame(columns=["Date", "Time", "Response", "Classification"]).to_csv(HISTORY_FILE, index=False)
    except Exception as e:
        # Fail gracefully; Streamlit will show the message later where appropriate
        pass

st.set_page_config(page_title="Competence Sentiment Analyzer", layout="wide")

# Load model & vectorizer (cached)
model, vectorizer = load_model_and_vectorizer()

# Background image base64 (optional)
bg_base64 = get_base64_of_file(BACKGROUND_IMAGE_PATH)

# Inject CSS (keeps your original styling, but now safe for deployment)
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: {LIGHT_BG} !important;
}}
.stApp {{
    background-color: transparent !important;
}}

body, [data-testid="stAppViewContainer"] {{
    font-family: 'Arial', sans-serif;
    color: {DARK_PRIMARY} !important;
}}

.block-container {{
    padding-top: 0rem !important;
    margin-top: 0rem !important;
    background-color: transparent !important;
}}

.home-logo-container img {{
    margin-top: 0px !important;
    margin-bottom:-50px !important;
    padding: 0px !important;
}}

[data-testid="stSidebar"] {{
    background: {DARK_PRIMARY} !important;
    padding: 5px 5px 50px 5px !important;
    display: flex;
    flex-direction: column;
    align-items: center;
}}

[data-testid="stSidebar"] img {{
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 55% !important;
    max-width: 50px;
    padding-top: 10px;
    padding-bottom: 40px;
}}

[data-testid="stSidebar"] div.stButton > button {{
    background: transparent !important;
    border: none !important;
    font-size: 20px !important;
    color: #e7f3fa !important;
    width: 100%;
    text-align: center !important;
    padding: 8px 0px 8px 0px !important;
    line-height: 1.0;
}}

.active-sidebar-button > button {{
    background: rgba(255,255,255,0.15) !important;
}}

h1 {{
    font-size: 46px !important;
    color: {DARK_PRIMARY} !important;
    font-weight: 700;
    margin-top: -30px !important;
    margin-bottom: 5px;
    text-align: center;
}}

textarea, .stTextInput input {{
    background: white !important;
    border: 3px solid #c7d9e2 !important;
    border-radius: 12px !important;
    color: {DARK_PRIMARY} !important;
    font-size: 18px !important;
    padding: 14px !important;
}}

div.stButton > button {{
    background-color: {ACCENT_BLUE} !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    border: none !important;
}}

.formal-blended-table {{
    border-collapse: collapse;
    width: 100%;
    font-family: 'Arial', sans-serif;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    border-radius: 8px;
    overflow: hidden;
    margin-top: 20px;
}}
.formal-blended-table th, .formal-blended-table td {{
    border: 1px solid #c7d9e2;
    padding: 12px;
}}
.formal-blended-table th {{
    background-color: {ACCENT_BLUE} !important;
    color: white !important;
    text-align: center;
    font-weight: bold;
}}
</style>
""", unsafe_allow_html=True)

# Simple sidebar navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

if os.path.exists(SIDEBAR_LOGO_PATH):
    st.sidebar.image(SIDEBAR_LOGO_PATH, use_column_width=False)


def sidebar_button_with_active_state(label, target_page):
    class_name = "active-sidebar-button" if st.session_state.page == target_page else ""
    st.sidebar.markdown(f'<div class="{class_name}">', unsafe_allow_html=True)
    if st.sidebar.button(label, key=f"nav_{target_page}"):
        st.session_state.page = target_page
        st.experimental_rerun()
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

sidebar_button_with_active_state("Home", "Home")
sidebar_button_with_active_state("Summary", "Summary")
sidebar_button_with_active_state("History", "History")

page = st.session_state.page

# Home page
if page == "Home":
    # apply bg if provided
    if bg_base64:
        st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{bg_base64}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
            background-repeat: no-repeat !important;
        }}
        .block-container {{ background-image: none !important; background-color: transparent !important; }}
        </style>
        """, unsafe_allow_html=True)

    if os.path.exists(HOME_LOGO_PATH):
        col1, col2, col3 = st.columns([3, 1, 3])
        with col2:
            st.markdown('<div class="home-logo-container">', unsafe_allow_html=True)
            st.image(HOME_LOGO_PATH, width=100)
            st.markdown('</div>', unsafe_allow_html=True)

    st.title("Competence Sentiment Analyzer")
    st.markdown(f'<p style="font-size: 16.5px; color: {DARK_PRIMARY}; margin-top: -5px; margin-bottom: 18px; text-align: center;">Welcome! The app analyzes students’ self-assessments of their Python, Java, and C programming competence using English-language sentiments.</p>', unsafe_allow_html=True)

    st.markdown("<div class='big-label'>", unsafe_allow_html=True)
    comment = st.text_area("Enter your sentiments about your programming competence:")
    st.markdown("</div>", unsafe_allow_html=True)

    col_button, col_result = st.columns([1, 2])

    with col_button:
        if st.button("Analyze", use_container_width=True):
            if not comment.strip():
                st.session_state.result = "Please enter some text first."
            elif model is None or vectorizer is None:
                st.session_state.result = "Model is not loaded. Cannot perform analysis."
            else:
                try:
                    vectorized_comment = vectorizer.transform([comment])
                    prediction = model.predict(vectorized_comment)[0]

                    sentiment_labels = {0: "Weak Competence", 1: "Normal Competence", 2: "Strong Competence"}
                    st.session_state.result = sentiment_labels.get(prediction, "Unknown")

                    now = datetime.now()
                    pd.DataFrame({
                        "Date": [now.strftime("%Y-%m-%d")],
                        "Time": [now.strftime("%H:%M:%S")],
                        "Response": [comment],
                        "Classification": [st.session_state.result]
                    }).to_csv(HISTORY_FILE, mode="a", header=False, index=False)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    with col_result:
        if "result" in st.session_state:
            if st.session_state.result == "Please enter some text first.":
                st.warning(st.session_state.result)
            elif "Model is not loaded" in st.session_state.result:
                st.error(st.session_state.result)
            else:
                st.success(f"Result: **{st.session_state.result}**")

# Summary page
elif page == "Summary":
    st.subheader("Summary of Responses")

    def transparent_config_fixed():
        return {
            "config": {
                "background": "transparent",
                "axis": {
                    "domainColor": "#c7d9e2",
                    "gridColor": "#c7d9e2",
                    "tickColor": "#c7d9e2",
                    "labelColor": DARK_PRIMARY,
                    "titleColor": DARK_PRIMARY,
                },
                "header": {
                    "titleColor": DARK_PRIMARY,
                    "labelColor": DARK_PRIMARY,
                }
            }
        }

    alt.themes.register("transparent_bg_fixed", transparent_config_fixed)
    alt.themes.enable("transparent_bg_fixed")

    if os.path.exists(HISTORY_FILE):
        df_history = pd.read_csv(HISTORY_FILE)
        if not df_history.empty:
            sentiment_counts = df_history['Classification'].value_counts().reset_index()
            sentiment_counts.columns = ['Competence Level', 'Count']
            order = ["Strong Competence", "Normal Competence", "Weak Competence"]
            df_full = pd.DataFrame({'Competence Level': order})
            sentiment_counts = pd.merge(df_full, sentiment_counts, on='Competence Level', how='left').fillna(0)
            sentiment_counts['Count'] = sentiment_counts['Count'].astype(int)
            sentiment_counts['Competence Level'] = pd.Categorical(sentiment_counts['Competence Level'], categories=order, ordered=True)
            sentiment_counts = sentiment_counts.sort_values('Competence Level')

            chart = alt.Chart(sentiment_counts).mark_bar().encode(
                x=alt.X('Competence Level', sort=order, axis=alt.Axis(title=None, labelAngle=0)),
                y=alt.Y('Count', title="Number of Responses"),
                color=alt.value(ACCENT_BLUE),
                tooltip=['Count']
            ).properties(title="").interactive()

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No responses recorded yet. Submit a response on the Home page.")
    else:
        st.error("Error: History file not found.")

# History page
elif page == "History":
    st.subheader("History of Responses")
    if os.path.exists(HISTORY_FILE):
        df_history = pd.read_csv(HISTORY_FILE)
        if not df_history.empty:
            df_history['DateTime'] = pd.to_datetime(df_history['Date'] + ' ' + df_history['Time'])
            df_history = df_history.sort_values(by='DateTime', ascending=False).drop(columns=['DateTime'])

            st.markdown("""
            <style>
            .formal-table { border-collapse: collapse; width: 100%; font-family: 'Arial', sans-serif; }
            .formal-table th, .formal-table td { border: 1px solid #c7d9e2; padding: 12px; }
            .formal-table th { background-color: #1f7fc1; color: white; text-align: left; }
            .formal-table td { background-color: #ffffff; }
            .formal-table tr:nth-child(even) td { background-color: #f1f7fb; }
            </style>
            """, unsafe_allow_html=True)

            st.markdown(df_history.to_html(index=False, classes="formal-table"), unsafe_allow_html=True)
        else:
            st.info("No sentiment history available.")
    else:
        st.info("History file not found.")

    col_delete_button, col_delete_message = st.columns([1, 3])
    with col_delete_button:
        if st.button("Delete History", use_container_width=True):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
                st.session_state.history_deleted = True
                st.experimental_rerun()
            else:
                st.info("No history to delete.")

    with col_delete_message:
        if st.session_state.get("history_deleted", False):
            st.success("History successfully deleted.")
            st.session_state.history_deleted = False

# hide menu
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
