import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import altair as alt
import time
from streamlit_lottie import st_lottie
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="SPAM GUARD AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Advanced Styling ---
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #F0F2F6;
    }

    /* Custom Cards */
    .card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease-in-out;
        height: 100%;
    }
    .card:hover {
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transform: translateY(-5px);
    }
    .card-title {
        font-size: 1.5em;
        font-weight: bold;
        color: #1E3A8A; /* Dark Blue */
        margin-bottom: 15px;
    }

    /* Custom Button Style */
    .stButton>button {
        border-radius: 10px;
        border: 2px solid #1E3A8A;
        background-color: #1E3A8A;
        color: white;
        padding: 12px 28px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: white;
        color: #1E3A8A;
        border-color: #1E3A8A;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #FFFFFF;
        border-right: 2px solid #E5E7EB;
    }

    /* Main Title Styling */
    .main-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: bold;
        color: #1E3A8A;
        padding: 20px 0;
    }
    .sub-title {
        text-align: center;
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 30px;
    }
    
    /* Result styling */
    .result-safe {
        background-color: #D1FAE5; border-left: 10px solid #10B981; padding: 20px; border-radius: 10px;
    }
    .result-spam {
        background-color: #FEE2E2; border-left: 10px solid #EF4444; padding: 20px; border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- Lottie Animation Loader ---
def load_lottieurl(url: str):
    """Fetches a Lottie animation from a URL."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.exceptions.RequestException:
        return None

# --- Load Animations ---
lottie_hero = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_t24tpvcu.json")
lottie_spam = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_f1cFs1.json") 
lottie_ham = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_R0sOCJ.json")
lottie_metrics = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_zlr9it5o.json")


# --- Data Caching & Model Training ---
@st.cache_data
def load_and_prepare_data(file_path):
    """Loads, prepares, and splits the dataset."""
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
        X, y = df['message'], df['label_num']
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except FileNotFoundError:
        st.error(f"Error: Dataset '{file_path}' not found.")
        st.info("Please ensure 'balanced_spam.csv' is in the application's root directory.")
        return None, None, None, None
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return None, None, None, None

@st.cache_resource
def train_model(X_train, y_train):
    """Trains the TF-IDF vectorizer and the classifier."""
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    classifier = PassiveAggressiveClassifier(max_iter=50, random_state=42)
    classifier.fit(X_train_tfidf, y_train)
    return vectorizer, classifier

# --- Load Data and Train ---
X_train, X_test, y_train, y_test = load_and_prepare_data('balanced_spam.csv')

# --- Page Rendering Functions ---

def show_home_page():
    """Renders the Home page."""
    st.markdown("<h1 class='main-title'>SPAM GUARD AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Your advanced shield against malicious and unwanted messages.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("""
        <div class='card'>
            <h2 class='card-title'>Welcome to the Future of Inbox Security</h2>
            <p>Spam Guard AI leverages a sophisticated Machine Learning model to provide real-time analysis of your messages. Our mission is to create a safer digital communication environment for everyone.</p>
            <br>
            <p>Use the navigation on the left to explore the tool's capabilities.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if lottie_hero:
            st_lottie(lottie_hero, height=350, key="hero_anim")

    st.markdown("---")
    st.markdown("<h2 class='card-title' style='text-align:center;'>Core Technology Stack</h2>", unsafe_allow_html=True)
    
    cols = st.columns(4)
    tech_stack = {
        "📊": "Data Preparation",
        "🔍": "Feature Extraction (TF-IDF)",
        "🧠": "Model Training (Passive-Aggressive)",
        "💡": "Real-time Prediction",
    }
    for i, (icon, tech) in enumerate(tech_stack.items()):
        with cols[i]:
            st.markdown(f"<div class='card'><p style='text-align:center; font-size: 3rem;'>{icon}</p><h3 style='text-align:center;'>{tech}</h3></div>", unsafe_allow_html=True)


def show_detector_page(vectorizer, model):
    """Renders the Spam Detector page."""
    st.markdown("<h1 class='main-title'>Real-Time Spam Detector</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        user_input = st.text_area("Enter message content here for analysis:", height=300, placeholder="Example: Congratulations! You've won a $1,000 gift card. Click here to claim...")
        if st.button("Analyze Message"):
            if user_input:
                with st.spinner('AI is analyzing the message...'):
                    time.sleep(1)
                    input_tfidf = vectorizer.transform([user_input])
                    prediction = model.predict(input_tfidf)
                
                st.markdown("---")
                if prediction[0] == 1:
                    st.markdown("<div class='result-spam'><h2 class='card-title'>Result: High Probability of SPAM 🚨</h2><p>Our analysis indicates this message is likely spam. Please exercise extreme caution and avoid clicking any links or providing personal information.</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='result-safe'><h2 class='card-title'>Result: Likely a Safe Message ✅</h2><p>This message appears to be legitimate. However, always remain vigilant when dealing with unexpected communications.</p></div>", unsafe_allow_html=True)

            else:
                st.warning("Please enter a message for analysis.")
    with col2:
        if 'prediction' in locals():
            anim = lottie_spam if prediction[0] == 1 else lottie_ham
            if anim:
                st_lottie(anim, height=300, key="result_anim")
        else:
             st.markdown("<div class='card'><h3 class='card-title'>Awaiting Input</h3><p>Your analysis result and a corresponding animation will appear here once you submit a message.</p></div>", unsafe_allow_html=True)

def show_insights_page(vectorizer, model, X_test_data, y_test_data):
    """Renders the Model Insights page."""
    st.markdown("<h1 class='main-title'>Model Performance Dashboard</h1>", unsafe_allow_html=True)
    
    # Model Evaluation
    X_test_tfidf = vectorizer.transform(X_test_data)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test_data, y_pred)
    cm = confusion_matrix(y_test_data, y_pred)
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
         st.markdown(f"<div class='card'><h2 class='card-title'>Model Accuracy</h2><p style='font-size: 3rem; font-weight: bold; color: #1E3A8A; text-align: center;'>{accuracy*100:.2f}%</p><p style='text-align: center;'>of messages on the test set were classified correctly.</p></div>", unsafe_allow_html=True)
         if lottie_metrics:
            st_lottie(lottie_metrics, height=250, key="metrics_anim")
    with col2:
        st.markdown("<div class='card'><h2 class='card-title'>Confusion Matrix</h2><p>This matrix provides a detailed breakdown of the model's prediction performance, showing correct vs. incorrect classifications.</p></div>", unsafe_allow_html=True)
        cm_df = pd.DataFrame(cm, index=['Actual: Ham', 'Actual: Spam'], columns=['Predicted: Ham', 'Predicted: Spam'])
        
        heatmap = alt.Chart(cm_df.stack().reset_index(name='value')).mark_rect(stroke='white', strokeWidth=1).encode(
            x=alt.X('level_1:O', title="Predicted Label", axis=alt.Axis(labelAngle=0)),
            y=alt.Y('level_0:O', title="Actual Label"),
            color=alt.Color('value:Q', scale=alt.Scale(scheme='viridis'), title="Count"),
            tooltip=['level_0', 'level_1', 'value']
        ).properties(width=400, height=300, title="Prediction vs. Actual Breakdown")
        
        text = heatmap.mark_text(baseline='middle', fontSize=16).encode(text='value:Q', color=alt.condition(alt.datum.value > 500, alt.value('white'), alt.value('black')))
        st.altair_chart(heatmap + text, use_container_width=True)


# --- Main Application Logic ---
# Sidebar Navigation
with st.sidebar:
    st.markdown("<h1 style='font-size: 24px; color: #1E3A8A;'>🛡️ SPAM GUARD AI</h1>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation Menu",
        ["🏠 Home", "🔎 Real-Time Detector", "📊 Performance Dashboard"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.info("This application is a demonstration of a machine learning model for spam detection.")

# Page routing
if X_train is not None:
    vectorizer, classifier = train_model(X_train, y_train)
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔎 Real-Time Detector":
        show_detector_page(vectorizer, classifier)
    elif page == "📊 Performance Dashboard":
        show_insights_page(vectorizer, classifier, X_test, y_test)
else:
    st.error("Application failed to start due to data loading issues. Please check the file path and data integrity.")

