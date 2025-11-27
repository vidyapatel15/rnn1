import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# Set page configuration
st.set_page_config(
    page_title="IMDB Movie Review Classifier by vidya",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .review-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #FF4B4B;
    }
    .positive {
        color: #00D100;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .negative {
        color: #FF4B4B;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-fill {
        background-color: #4CAF50;
        height: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        line-height: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">IMDB Movie Review Classifier by vidya</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model_and_word_index():
    """Load model and word index with error handling"""
    try:
        # Load word index
        word_index = tf.keras.datasets.imdb.get_word_index()
        
        # Try to load pre-trained model, if not exists, create a simple one
        try:
            model = tf.keras.models.load_model('rnn_movie_review_model.h5')
            st.sidebar.success("✅ Pre-trained model loaded!")
        except:
            st.sidebar.info("🔄 No pre-trained model found. Using a simple model for demo...")
            # Create a simple model for demo purposes
            model = create_simple_model()
        
        return model, word_index
    except Exception as e:
        st.error(f"Error during initialization: {e}")
        return None, {}

def create_simple_model():
    """Create a simple model for demo purposes"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
    
    model = Sequential([
        Embedding(10000, 100, input_length=500),
        SimpleRNN(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def decode_review(encoded_review, word_index):
    """Decode encoded review back to text"""
    reverse_word_index = {value: key for key, value in word_index.items()}
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review if i >= 3])

def preprocess_text(text, word_index, vocab_size=10000, max_length=500):
    """Preprocess user input text"""
    words = text.lower().split()
    encoded = []
    for word in words:
        if word in word_index and word_index[word] < vocab_size - 3:
            encoded.append(word_index[word] + 3)
        else:
            encoded.append(2)  # <UNK> token
    encoded = pad_sequences([encoded], maxlen=max_length, padding='post', truncating='post')
    return encoded

# Initialize app
model, word_index = load_model_and_word_index()

# Sample reviews from IMDB dataset
sample_reviews_encoded = [
    [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4],
    [1, 194, 1153, 194, 8255, 78, 228, 5, 6, 1463, 4369, 5012, 134, 26, 4, 715, 8, 118, 1634, 14, 394, 20, 13, 119, 954, 189, 102, 5, 207, 110, 3103, 21, 14, 69, 188, 8, 30, 23, 7, 4, 249, 126, 93, 4, 114, 9, 2215, 2197, 29, 179, 412, 8, 232, 385, 1090, 72, 1127, 43, 101, 14],
    [1, 785, 189, 438, 47, 110, 142, 7, 6, 432, 2128, 197, 2869, 418, 134, 42, 500, 21, 656, 13, 4, 1766, 1942, 5, 121, 28, 239, 122, 60, 13, 258, 99, 102, 11, 6, 124, 36, 119, 179, 19, 14, 715, 953, 59, 13, 258, 239, 122, 60, 813, 14, 12, 18, 316, 69, 114, 11, 171, 208],
    [1, 67, 59, 128, 74, 12, 716, 4, 119, 11, 2209, 1204, 8, 2054, 10, 10, 1361, 4, 118, 14, 394, 12, 133, 134, 9, 1442, 3394, 12, 3041, 1093, 4, 173, 13, 447, 6167, 147, 8, 365, 4, 714, 453, 739, 209, 11, 2209, 1204, 8, 2054, 10, 10, 13, 258, 239, 122, 60, 813, 14, 12],
    [1, 778, 128, 74, 12, 630, 163, 15, 4, 1766, 7982, 1051, 2, 32, 85, 156, 45, 40, 148, 139, 121, 664, 665, 10, 10, 1361, 4, 118, 14, 394, 12, 133, 134, 9, 1442, 3394, 12, 3041, 1093, 4, 173, 13, 447, 6167, 147, 8, 365, 4, 714, 453, 739, 209, 11, 2209, 1204, 8, 2054]
]

# Sidebar for user input
st.sidebar.header("Custom Review Classification")
user_review = st.sidebar.text_area("Enter your movie review:", height=150, 
                                  placeholder="Type your movie review here...")

if st.sidebar.button("Classify Review"):
    if user_review.strip() and model is not None:
        with st.spinner("Classifying review..."):
            try:
                # Preprocess and predict
                processed_review = preprocess_text(user_review, word_index)
                prediction = model.predict(processed_review, verbose=0)[0][0]
                sentiment = "POSITIVE" if prediction > 0.5 else "NEGATIVE"
                confidence = prediction if prediction > 0.5 else 1 - prediction
                
                st.sidebar.success(f"Prediction: **{sentiment}**")
                st.sidebar.info(f"Confidence: **{confidence:.2%}**")
                
                # Display confidence bar
                confidence_percent = int(confidence * 100)
                st.sidebar.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence_percent}%;">
                        {confidence_percent}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.sidebar.error(f"Error during classification: {e}")
    else:
        st.sidebar.warning("Please enter a review to classify.")

# Main content
st.header("Sample Movie Reviews Classification")

if model is not None and word_index:
    st.write("Below are 5 sample movie reviews from the IMDB dataset and their classification results:")
    
    for i, encoded_review in enumerate(sample_reviews_encoded, 1):
        # Decode the review
        decoded_review = decode_review(encoded_review, word_index)
        
        # Prepare the review for prediction
        processed_review = pad_sequences([encoded_review], maxlen=500, padding='post', truncating='post')
        
        # Make prediction
        try:
            prediction = model.predict(processed_review, verbose=0)[0][0]
            sentiment = "POSITIVE" if prediction > 0.5 else "NEGATIVE"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            # Display results
            st.markdown(f"### Review {i}")
            st.markdown(f'<div class="review-box">{decoded_review}</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if sentiment == "POSITIVE":
                    st.markdown(f'<p class="positive">🎬 Sentiment: {sentiment}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="negative">🎬 Sentiment: {sentiment}</p>', unsafe_allow_html=True)
            
            with col2:
                st.write(f"**Confidence: {confidence:.2%}**")
                
                # Confidence bar
                confidence_percent = int(confidence * 100)
                st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence_percent}%;">
                        {confidence_percent}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
        except Exception as e:
            st.error(f"Error processing sample review {i}: {e}")

else:
    st.error("Model initialization failed. Please check the logs for details.")

# Model info in sidebar
st.sidebar.markdown("---")
st.sidebar.header("Model Information")
st.sidebar.info("""
**Model Type:** Simple RNN  
**Architecture:**
- Embedding Layer (10,000 vocab)
- SimpleRNN Layer (64 units)
- Dense Layer (32 units)
- Output Layer (sigmoid)

**Training Data:** IMDB Movie Reviews
**Task:** Binary Sentiment Classification
""")

# Footer
st.markdown("---")
st.markdown("### Built with Streamlit and TensorFlow | RNN Movie Review Classifier")
st.markdown("*Note: This model classifies movie reviews as POSITIVE or NEGATIVE based on the IMDB dataset.*")
