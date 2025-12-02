import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import base64
import altair as alt
import sys

MODEL_PATH = "C:/Users/Sherylle Rose/Desktop/rfmodeloct26/rf_model_final.pkl"
VECTORIZER_PATH = "C:/Users/Sherylle Rose/Desktop/rfmodeloct26/vectorizer_final.pkl"

HOME_LOGO_PATH = "C:/Users/Sherylle Rose/homeplogo.png"
SIDEBAR_LOGO_PATH = "C:\\Users\\Sherylle Rose\\ccitlogo.png"
BACKGROUND_IMAGE_PATH = "C:\\Users\\Sherylle Rose\\bg_final.png"

LIGHT_BG = "#e8f3f8"
DARK_PRIMARY = "#1f3a52"
ACCENT_BLUE = "#1f7fc1"

def get_base64_of_file(path):
    try:
        if not os.path.exists(path):
            st.error(f"Error: Background file not found at: {path}. Please check the path and run the script again.")
            return None
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.error(f"Error encoding background file: {e}")
        return None

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError:
    st.error("Error: Model or vectorizer file not found. Please check paths.")
    model = None
    vectorizer = None

bg_base64 = get_base64_of_file(BACKGROUND_IMAGE_PATH)

HISTORY_FILE = "sentiment_history.csv"
if not os.path.exists(HISTORY_FILE):
    try:
        pd.DataFrame(columns=["Date", "Time", "Response", "Classification"]).to_csv(HISTORY_FILE, index=False)
    except Exception as e:
        st.error(f"Critical initialization error: Could not create empty history file. Error: {e}")

st.set_page_config(page_title="Competence Sentiment Analyzer", layout="wide")

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

.css-18e3th9, .css-1d391kg, .block-container,
[data-testid="stVerticalBlock"] > div:first-child > div:first-child {{
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}}

.home-logo-container img {{
    margin-top: 0px !important;
    margin-bottom:-50px !important;
    padding: 0px !important;
}}

[data-testid="stSidebar"] {{
    background: {DARK_PRIMARY} !important;
    width: 20px !important;
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

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0px;
}}

[data-testid="stSidebarNavItems"] {{
    width: 100%;
    align-items: center !important;
    padding-left: 0px !important;
}}

[data-testid="stSidebar"] div.stButton > button {{
    background: transparent !important;
    border: none !important;
    font-size: 30px !important;
    color: #e7f3fa !important;
    width: 100%;

    max-width: none; /* Make button full width of sidebar's content area */
    margin-right: auto;
    margin-left: auto !important;

    text-align: center !important; /* Center the button text */
    padding: 10px 0px 10px 0px !important;
    line-height: 1.0;
    display: flex !important;
    justify-content: center !important; /* Center the button text */
}}

[data-testid="stSidebar"] div.stButton > button:hover {{
    background: rgba(255,255,255,0.15) !important;
    border-radius: 0px !important;
    width: 100% !important; /* Ensure hover is full length of sidebar content */
}}

.active-sidebar-button > button {{
    background: rgba(255,255,255,0.15) !important;
    border-radius: 0px !important;
    width: 100% !important; /* Ensure active state is full length of sidebar content */
}}

h1 {{
    font-size: 55px !important;
    color: {DARK_PRIMARY} !important;
    font-weight: 700;
    margin-top: -50px !important;
    margin-bottom: 5px;
    text-align: center;
}}
h2 {{ font-size: 32px !important; color: {DARK_PRIMARY} !important; }}
h3 {{ font-size: 28px !important; color: {DARK_PRIMARY} !important; }}

textarea, .stTextInput input {{
    background: white !important;
    border: 3px solid #c7d9e2 !important;
    border-radius: 12px !important;
    color: {DARK_PRIMARY} !important;
    font-size: 18px !important;
    padding: 14px !important;
}}

textarea:focus, textarea:hover,
.stTextInput input:focus, .stTextInput input:hover {{
    border: 3px solid {DARK_PRIMARY} !important;
    color: {DARK_PRIMARY} !important;
    outline: none !important;
}}

[data-testid="stTextarea"] > div > div:focus,
[data-testid="stTextarea"] > div > div:focus-within,
[data-testid="stTextarea"] > div > div:hover {{
    border-color: {DARK_PRIMARY} !important;
    box-shadow: none !important;
}}
textarea:focus {{
    border: 3px solid {DARK_PRIMARY} !important;
}}

.big-label label {{
    font-size: 40px !important;
    font-weight: 700 !important;
    color: {DARK_PRIMARY} !important;
    margin-bottom: 10px;
}}

table, th, td {{ color: {DARK_PRIMARY} !important; }}
th {{ background: {ACCENT_BLUE} !important; color: white !important; }}
</style>

<style>

div.stButton > button {{
    background-color: {ACCENT_BLUE} !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    border: none !important;
    transition: background 0.3s ease;
}}
div.stButton > button:hover {{
    background-color: #155f8a !important;
}}

.stAlert {{
    width: 100% !important;
    margin-left: 0px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 8px !important;
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
    color: {DARK_PRIMARY} !important;
}}
.formal-blended-table th {{
    background-color: {ACCENT_BLUE} !important;
    color: white !important;
    text-align: center;
    font-weight: bold;
}}
.formal-blended-table td {{
    background-color: rgba(255, 255, 255, 0.8) !important;
    text-align: center;
}}
.formal-blended-table tr:nth-child(even) td {{
    background-color: rgba(241, 247, 251, 0.8) !important;
}}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "Home"

if os.path.exists(SIDEBAR_LOGO_PATH):
    st.sidebar.image(SIDEBAR_LOGO_PATH)

page = st.session_state.page

def sidebar_button_with_active_state(label, target_page, icon):
    class_name = "active-sidebar-button" if st.session_state.page == target_page else ""

    st.sidebar.markdown(f'<div class="{class_name}">', unsafe_allow_html=True)

    if st.sidebar.button(label, key=f"nav_{target_page}"):
        st.session_state.page = target_page
        st.rerun()

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

sidebar_button_with_active_state("Home", "Home", "")
sidebar_button_with_active_state("Summary", "Summary", "")
sidebar_button_with_active_state("History", "History", "")

if page == "Home":

    if bg_base64:
        home_page_css = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{bg_base64}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
            background-repeat: no-repeat !important;
        }}
        .block-container {{
            background-image: none !important;
            background-color: transparent !important;
            overflow-y: hidden !important;
        }}
        </style>
        """
        st.markdown(home_page_css, unsafe_allow_html=True)

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
                vectorized_comment = vectorizer.transform([comment])
                prediction = model.predict(vectorized_comment)[0]

                sentiment_labels = {
                    0: "Weak Competence",
                    1: "Normal Competence",
                    2: "Strong Competence"
                }
                st.session_state.result = sentiment_labels.get(prediction, "Unknown")

                now = datetime.now()
                pd.DataFrame({
                    "Date": [now.strftime("%Y-%m-%d")],
                    "Time": [now.strftime("%H:%M:%S")],
                    "Response": [comment],
                    "Classification": [st.session_state.result]
                }).to_csv(HISTORY_FILE, mode="a", header=False, index=False)

    with col_result:
        if "result" in st.session_state:
            if st.session_state.result == "Please enter some text first.":
                st.warning(st.session_state.result)
            elif "Model is not loaded" in st.session_state.result:
                st.error(st.session_state.result)
            else:
                st.success(f"Result: **{st.session_state.result}**")

elif page == "Summary":

    st.markdown("""
    <style>
    .block-container {
        background-image: none !important;
        background-color: transparent !important;
        overflow-y: hidden !important; /* Forces non-scrollable for Summary page */
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("Summary of Responses")

    st.markdown("<br><br>", unsafe_allow_html=True)

    def transparent_config_fixed():
        return {
            "config": {
                "background": "transparent",
                "view": {
                    "fill": "transparent",
                    "stroke": "transparent"
                },

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

            sentiment_counts = pd.merge(
                df_full,
                sentiment_counts,
                on='Competence Level',
                how='left'
            ).fillna(0)

            sentiment_counts['Count'] = sentiment_counts['Count'].astype(int)

            sentiment_counts['Competence Level'] = pd.Categorical(
                sentiment_counts['Competence Level'], categories=order, ordered=True
            )
            sentiment_counts = sentiment_counts.sort_values('Competence Level')

            chart = alt.Chart(sentiment_counts).mark_bar().encode(
                x=alt.X('Competence Level', sort=order, axis=alt.Axis(title=None, labelAngle=0)),
                y=alt.Y('Count', title="Number of Responses"),
                color=alt.value(ACCENT_BLUE),
                tooltip=['Count']
            ).properties(
                title=""
            ).interactive()

            st.altair_chart(chart, use_container_width=True)

        else:
            st.info("No responses recorded yet. Submit a response on the Home page.")
    else:
        st.error("Error: History file not found.")

elif page == "History":

    st.markdown("""
    <style>
    .block-container {
        background-image: none !important;
        background-color: transparent !important;
        overflow-y: auto !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("History of Responses")

    if os.path.exists(HISTORY_FILE):
        df_history = pd.read_csv(HISTORY_FILE)

        if not df_history.empty:
            # Sort the DataFrame by Date and Time in descending order (latest first)
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
                st.rerun()
            else:
                st.info("No history to delete.")

    with col_delete_message:
        if st.session_state.get("history_deleted", False):
            st.success("History successfully deleted.")
            st.session_state.history_deleted = False

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)