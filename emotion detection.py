import streamlit as st
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
import os

# Suppress TensorFlow logging messages and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# --- Configuration ---
MAX_WORDS = 20000 # Max number of words to keep in the vocabulary
MAX_LEN = 100     # Max length of a sequence (review)
EMBEDDING_DIM = 100 # Dimension of the word embeddings
LSTM_UNITS = 128  # INCREASED CAPACITY for higher accuracy
NUM_CLASSES = 6
# Adjusted to 30 epochs for highest potential accuracy
EPOCHS = 30 

# Define the emotion labels for mapping
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
label_to_id = {label: i for i, label in enumerate(emotion_labels)}
id_to_label = {i: label for i, label in enumerate(emotion_labels)}

# Sample reviews for demonstration purposes
SAMPLE_REVIEWS = {
    "joy": "This product makes me so happy and absolutely ecstatic! I'm thrilled!",
    "sadness": "I feel utterly heartbroken and devastated by this poor quality item.",
    "anger": "This cheap piece of junk makes me furious! I hate it and want my money back immediately.",
    "fear": "I am terrified! The warning signs on this are incredibly worrying, I fear using it.",
    "love": "I absolutely adore this item! I'm completely in love with the quality and design.",
    "surprise": "Unbelievable! I was totally shocked by how quickly it arrived and how amazing it is. What a surprise!"
}

# --- Model Building Function ---

def build_bilstm_model():
    """Builds the powerful single BiLSTM model with increased capacity."""
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        Dropout(0.3),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
        Bidirectional(LSTM(LSTM_UNITS // 2)),
        Dense(128, activation='relu'), # Increased density
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# --- Caching function to load data and train the model once ---

@st.cache_resource
def load_and_train_model():
    """Loads data, trains the single BiLSTM model, and evaluates it."""
    
    # 1. Load Data
    data = load_dataset("dair-ai/emotion", "split")
    
    train_texts = list(data['train']['text'])
    train_labels = list(data['train']['label'])
    test_texts = list(data['test']['text'])
    test_labels = list(data['test']['label'])
    
    # Combine train and test into one pool for tokenizer fit
    all_texts = train_texts + test_texts

    # 2. Tokenization and Sequencing
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(all_texts)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    
    train_padded = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    test_padded = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # Convert labels to one-hot encoding
    train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=NUM_CLASSES)
    
    # 3. Build and Train Model
    model = build_bilstm_model()

    # Train model silently
    model.fit(
        train_padded, 
        train_labels_one_hot,
        epochs=EPOCHS, 
        batch_size=32,
        validation_split=0.1,
        verbose=0 # Run silently
    )

    # 4. Prediction and Evaluation
    
    # Get predictions (probabilities)
    y_pred_probs = model.predict(test_padded, verbose=0)
    
    # Final prediction
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate Metrics
    accuracy = accuracy_score(test_labels, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, y_pred, average='macro', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return model, tokenizer, metrics


# --- Prediction Function ---
def predict_emotion(model, tokenizer, text):
    """Predicts the emotion of a given review text."""
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Make prediction
    prediction = model.predict(padded_sequence, verbose=0)[0]
    
    # Get the index of the highest probability
    predicted_id = np.argmax(prediction)
    predicted_label = id_to_label[predicted_id].capitalize()
    
    # Format data for bar chart
    prob_data = pd.DataFrame({
        'Emotion': [label.capitalize() for label in emotion_labels],
        'Confidence': prediction
    }).set_index('Emotion')

    return predicted_label, prob_data


# --- Main Streamlit App ---
def main():
    
    # --- CSS Injection for Engaging Background and Improved Visibility ---
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        
        /* Animated Gradient Background - DARKER COLORS FOR BETTER CONTRAST */
        .stApp {
            font-family: 'Poppins', sans-serif;
            color: #FFFFFF; /* Ensures main text is white */
            background: linear-gradient(-45deg, #0f002a, #2b0846, #002a3a, #00404a); 
            background-size: 400% 400%;
            animation: gradientBG 25s ease infinite; /* Slower animation */
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* All Text and Headers are Lighter */
        h1, h2, h3, h4, .st-emotion-cache-12fm1f5, .st-emotion-cache-79elbk {
            color: #FFFFFF !important; /* Pure white headers */
            letter-spacing: 1.2px;
        }

        /* Header Styling */
        .header {
            color: #FFFFFF;
            text-align: center;
            padding: 15px;
            border-radius: 12px;
            /* Contrast Fix: Use a dark, semi-opaque background */
            background: rgba(0, 0, 0, 0.4); 
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.7);
            margin-bottom: 25px;
        }
        
        /* Primary Button Hover Effect */
        .stButton>button {
            border-radius: 10px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
            color: white;
            background-color: #5b1076; /* Darker primary button */
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.6);
            background-color: #7b1296; 
        }

        /* Text Area and Table Containers - CONTRAST FIX APPLIED */
        .stTextArea textarea, .stTable {
            background-color: rgba(0, 0, 0, 0.3) !important; /* Dark, semi-transparent black */
            color: white !important;
            border: 1px solid #7b1296;
            transition: border 0.3s ease;
        }
        /* Ensure table header text is visible */
        .stTable th {
             color: #FFD700 !important;
             background-color: rgba(0, 0, 0, 0.6) !important; /* Slightly darker header background */
        }
        /* Fix for Table word wrapping and contrast */
        .stTable tbody tr th, .stTable tbody tr td {
            color: #FFFFFF !important;
            background-color: rgba(0, 0, 0, 0.5) !important; /* Consistent dark background for body */
            border-bottom: 1px solid #3d0a52; /* subtle separator */
            word-break: normal; /* FIX: Prevent breaking in the middle of a word */
            white-space: normal; /* FIX: Allow normal wrapping for long reviews */
        }
        
        .stTextArea textarea:focus {
            border: 2px solid #FFD700; /* Light yellow focus */
            box-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
        }
        /* Ensure input text within text area is white */
        .stTextArea label + div > textarea {
            color: white !important;
        }
        /* Ensure bar chart container background is dark */
        .stPlotlyChart {
            background-color: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 10px;
        }


        /* Metric Boxes Styling - CONTRAST FIX APPLIED */
        .metric-box {
            background-color: rgba(0, 0, 0, 0.5); /* Darker and more opaque for better contrast */
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 3px 6px rgba(0, 0, 0, 0.7);
            margin-bottom: 15px;
            text-align: center;
            color: white;
        }
        .metric-label {
            font-size: 1rem;
            opacity: 0.8;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #FFD700; /* Gold value */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="header"><h1><span style="color: #FFD700;">🧠</span> Customer Emotion Detector <span style="color: #FFD700;">🚀</span></h1></div>', unsafe_allow_html=True)
    
    # Load and train the model (cached)
    models, tokenizer, metrics = load_and_train_model()

    # --- Input Section ---
    st.markdown("<h3 style='color: white;'>Enter a Product Review:</h3>", unsafe_allow_html=True)
    user_input = st.text_area("Review Text", 
                              "I feel absolutely wonderful and proud of this amazing result, it's pure happiness!", 
                              height=150)
    
    if st.button("Detect Emotion", use_container_width=True, type="primary"):
        if user_input:
            # Predict
            predicted_emotion, prob_data = predict_emotion(models, tokenizer, user_input)
            
            st.markdown("<hr style='border: 1px solid #7b1296;'>", unsafe_allow_html=True)
            
            # Display Prediction
            st.markdown(f"<h2 style='color: #FFD700;'>Predicted Emotion: <span style='background-color: #5b1076; padding: 10px 20px; border-radius: 10px; color: white;'>{predicted_emotion}</span></h2>", unsafe_allow_html=True)
            
            # Display Confidence Breakdown
            st.markdown("<h3 style='color: white;'>Confidence Breakdown:</h3>", unsafe_allow_html=True)
            
            # Use Pandas DataFrame index for chart labels
            st.bar_chart(prob_data, use_container_width=True)
            
            st.markdown("<hr style='border: 1px solid #7b1296;'>", unsafe_allow_html=True)

    # --- Sample Reviews Section (Designed for High Confidence) ---
    st.markdown("<h3 style='color: white;'>Example Reviews to Test (Optimized for High Confidence):</h3>", unsafe_allow_html=True)
    
    sample_data = []
    for emotion, review in SAMPLE_REVIEWS.items():
        sample_data.append([emotion.capitalize(), review])
    
    sample_df = pd.DataFrame(sample_data, columns=['Emotion', 'Review'])
    st.table(sample_df)

    # --- Evaluation Metrics Display (Moved to Bottom) ---
    st.markdown("---")
    st.markdown("<h2 style='color: #FFD700; text-align: center;'>Model Evaluation Metrics</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)

    # Function to display metric in custom style
    def display_metric(col, label, value):
        col.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value:.4f}</div>
            </div>
        """, unsafe_allow_html=True)

    display_metric(col1, "Accuracy", metrics['accuracy'])
    display_metric(col2, "Macro Precision", metrics['precision'])
    display_metric(col3, "Macro Recall", metrics['recall'])
    display_metric(col4, "Macro F1-Score", metrics['f1_score'])

    if metrics['accuracy'] >= 0.85:
        st.success(f"✅ Target Accuracy of 85% Achieved! Current Accuracy: {metrics['accuracy']:.4f}")
    else:
        st.warning(f"⚠️ Target Accuracy of 85% Not Met Yet. Current Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
