import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📱", layout="centered")

# --- NLTK DOWNLOADS ---
# Caching prevents Streamlit from downloading these on every UI interaction
@st.cache_resource
def download_nltk_dependencies():
    nltk.download('punkt')
    nltk.download('punkt_tab') # Required for newer versions of NLTK
    nltk.download('stopwords')

download_nltk_dependencies()
ps = PorterStemmer()

# --- PREPROCESSING FUNCTION ---
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# --- LOAD MODELS ---
# Caching ensures models are loaded into memory only once
@st.cache_resource
def load_models():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_models()

# --- UI DESIGN ---
st.title("📱 SMS Spam Classifier")
st.write("Detect whether an SMS or Email is **Spam** or **Not Spam (Ham)** using Natural Language Processing.")

st.markdown("### Test the Model")
example_selection = st.selectbox(
    "Choose a sample message or type your own:",
    [
        "Type your own message...", 
        "Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/12345 to claim now.", 
        "Hey, are we still meeting for lunch at 1 PM tomorrow? Let me know."
    ]
)

# Populate text area based on dropdown selection
if example_selection == "Type your own message...":
    input_sms = st.text_area("Enter the message you want to classify:", height=150)
else:
    input_sms = st.text_area("Enter the message you want to classify:", value=example_selection, height=150)

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to predict.")
    else:
        with st.spinner("Analyzing text..."):
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)
            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. Predict
            result = model.predict(vector_input)[0]
            
            # 4. Display Result
            if result == 1:
                st.error("🚨 **Alert! This message is classified as SPAM.**")
            else:
                st.success("✅ **Good news! This message is classified as NOT SPAM (Ham).**")

st.markdown("---")
with st.expander("ℹ️ How it works"):
    st.write('''
    This application processes text using a Machine Learning pipeline:
    1. **Text Cleaning**: Converts everything to lowercase, tokenizes sentences, and removes special characters.
    2. **Stop-word Removal & Stemming**: Removes common filler words and reduces words to their root form (e.g., 'dancing' -> 'danc').
    3. **Vectorization**: Transforms the cleaned text into numerical format using TF-IDF.
    4. **Classification**: Feeds the numbers into a trained Naive Bayes model to determine if the pattern matches a Spam or Ham class.
    ''')