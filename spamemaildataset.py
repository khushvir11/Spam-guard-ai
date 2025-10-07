import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import altair as alt
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Data Caching ---
# Cache the data loading and model training to avoid re-running on every interaction
@st.cache_data
def load_and_prepare_data(file_path):
    """Loads the balanced data and splits it for training and testing."""
    try:
        # Load the pre-cleaned and balanced dataset
        df = pd.read_csv(file_path, encoding='latin1')
        
        # Map labels to numbers
        df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
        X = df['message']
        y = df['label_num']
        
        # Split the data
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please make sure it's in the correct directory.")
        return None, None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None, None, None, None


@st.cache_resource
def train_model(X_train, y_train):
    """Trains the TF-IDF vectorizer and the classifier."""
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    
    pac = PassiveAggressiveClassifier(max_iter=50, random_state=42)
    pac.fit(tfidf_train, y_train)
    return tfidf_vectorizer, pac

# --- Load Data and Train Model ---
# The app now uses the balanced dataset
X_train, X_test, y_train, y_test = load_and_prepare_data('balanced_spam.csv')

if X_train is not None:
    tfidf_vectorizer, pac_model = train_model(X_train, y_train)

    # --- UI Elements ---
    st.markdown("<h1 style='text-align: center; color: #007BFF;'>📧 Email Spam Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Enter the content of an email or message below to check if it's spam or not.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Main Interaction ---
    with st.container():
        st.subheader("Check Your Message")
        user_input = st.text_area("Paste the email content here:", height=200, placeholder="Type or paste your message...")
        predict_button = st.button("Analyze Message", type="primary")

        if predict_button:
            if user_input:
                with st.spinner('Analyzing...'):
                    time.sleep(1) # Simulate processing time for better UX
                    
                    # Vectorize the user input
                    tfidf_input = tfidf_vectorizer.transform([user_input])
                    
                    # Make prediction
                    prediction = pac_model.predict(tfidf_input)
                    
                    st.markdown("---")
                    st.subheader("Analysis Result")
                    if prediction[0] == 1:
                        st.error("🚨 This looks like a SPAM message.")
                        st.image("https://placehold.co/600x100/f03e3e/ffffff?text=SPAM+ALERT&font=lato", use_column_width=True)
                    else:
                        st.success("✅ This looks like a legitimate message (HAM).")
                        st.image("https://placehold.co/600x100/40c057/ffffff?text=SAFE+MESSAGE&font=lato", use_column_width=True)
            else:
                st.warning("Please enter a message to analyze.")

    # --- Sidebar for Model Performance ---
    with st.sidebar:
        st.header("Model Performance")
        st.markdown("The model is trained on a **balanced** dataset of SMS messages to classify them as spam or ham.")
        
        # Evaluate model
        tfidf_test = tfidf_vectorizer.transform(X_test)
        y_pred = pac_model.predict(tfidf_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.metric(label="Model Accuracy", value=f"{accuracy*100:.2f}%")

        st.markdown("##### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Actual Ham', 'Actual Spam'], columns=['Predicted Ham', 'Predicted Spam'])
        
        # Create a heatmap using Altair
        heatmap = alt.Chart(cm_df.stack().reset_index(name='value')).mark_rect().encode(
            x=alt.X('level_1:O', title="Predicted Label"),
            y=alt.Y('level_0:O', title="Actual Label"),
            color=alt.Color('value:Q', scale=alt.Scale(scheme='blues'), title="Count"),
            tooltip=['level_0', 'level_1', 'value']
        ).properties(
            width=250,
            height=200,
            title="Confusion Matrix"
        )
        st.altair_chart(heatmap, use_container_width=True)

        st.markdown("---")
        st.info("This is a demo application. The accuracy reflects performance on the test portion of the uploaded dataset.")

# This message is shown if the file is not found
else:
     st.error("Failed to load the dataset. Please ensure 'balanced_spam.csv' is uploaded and accessible.")

