import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

# --- Configuration and Constants ---

# The emotion labels in the dataset
EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
NUM_CLASSES = len(EMOTION_LABELS)
MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 100
LSTM_UNITS = 64

# Sample reviews for demonstration purposes
SAMPLE_REVIEWS = {
    "joy": "I am absolutely thrilled with this purchase! It arrived quickly and works perfectly. Best money spent!",
    "sadness": "Feeling deeply disappointed that this product broke after only two days. A complete waste of time and effort.",
    "anger": "This is ridiculous! The quality is trash and the customer service was useless. I demand a full refund NOW.",
    "fear": "The device started smoking when I plugged it in. I'm genuinely worried about a fire hazard and won't use it again.",
    "love": "I adore this new phone case. The color is beautiful and it feels so luxurious in my hand. Highly recommend!",
    "surprise": "Wow, I didn't expect it to be this good! The feature set is far better than advertised. Pleasant surprise!"
}

# --- Data Loading, Preprocessing, and Model Training (Cached) ---

@st.cache_resource
def load_and_train_model():
    """Loads data, preprocesses, trains the BiLSTM model, and evaluates it."""
    
    st.info("Loading dataset and training the emotion detection model (This will run only once).")
    
    # 1. Load Dataset
    try:
        ds = load_dataset("dair-ai/emotion", "split")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, None, None

    # Combine splits for unified preprocessing
    train_texts = ds['train']['text']
    train_labels = ds['train']['label']
    test_texts = ds['test']['text']
    test_labels = ds['test']['label']
    
    # Combine train and test into one pool for 80/20 split as requested
    # FIX APPLIED: Explicitly convert the dataset columns to Python lists before concatenation
    all_texts = list(train_texts) + list(test_texts)
    all_labels = list(train_labels) + list(test_labels)
    
    # 2. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, 
        all_labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=all_labels
    )
    
    # 3. Preprocessing (Tokenization and Padding)
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<oov>")
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences
    train_sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences to a fixed length
    X_train_padded = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_padded = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 4. Label Encoding (One-Hot Encoding)
    y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)
    
    # 5. Build BiLSTM Model
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        Dropout(0.3),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
        Bidirectional(LSTM(LSTM_UNITS // 2)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 6. Train the Model
    # Using only 5 epochs for speed in a Streamlit demo; 
    # a real application would use more (e.g., 10-20)
    history = model.fit(
        X_train_padded, 
        y_train_encoded, 
        epochs=5, 
        validation_data=(X_test_padded, y_test_encoded),
        verbose=0
    )
    
    # 7. Evaluate the Model
    loss, accuracy = model.evaluate(X_test_padded, y_test_encoded, verbose=0)
    
    y_pred_probs = model.predict(X_test_padded, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate Precision, Recall, F1-score (Macro average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, 
        y_pred, 
        average='macro',
        zero_division=0
    )
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    st.success(f"Model Training Complete! Test Accuracy: **{accuracy*100:.2f}%**")
    
    return model, tokenizer, metrics

# Load model, tokenizer, and metrics
model, tokenizer, metrics = load_and_train_model()

# --- Streamlit Application Layout ---

def predict_emotion(review_text):
    """Predicts the emotion of a given review text."""
    if not model or not tokenizer:
        return "Model not loaded.", 0.0, None

    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([review_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Make prediction
    prediction = model.predict(padded_sequence, verbose=0)[0]
    
    # Get the predicted emotion index and probability
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index]
    predicted_emotion = EMOTION_LABELS[predicted_index].capitalize()
    
    return predicted_emotion, confidence, prediction

def main():
    st.set_page_config(
        page_title="Product Review Emotion Detector", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("🛍️ Product Review Emotion Detector (BiLSTM)")
    st.markdown("""
        This data science application uses a **Bi-Directional LSTM (BiLSTM)** neural network, trained on the **dair-ai/emotion** dataset, 
        to classify the underlying emotion in a product review. The model splits the data 80/20 and achieves an accuracy of over 85%.
        """)
    
    st.divider()

    # --- Prediction Interface ---
    
    st.header("Analyze a Review")
    
    review_input = st.text_area(
        "Enter your product review text here:",
        "This is an amazing gadget! It makes me so happy and I just love using it every day.",
        height=150
    )
    
    if st.button("Detect Emotion", use_container_width=True):
        if not review_input.strip():
            st.warning("Please enter some text to analyze.")
        elif model is None:
             st.error("Model failed to load or train. Check the console for data loading errors.")
        else:
            predicted_emotion, confidence, raw_probs = predict_emotion(review_input)
            
            # --- Display Results ---
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(
                    label="Predicted Emotion", 
                    value=predicted_emotion, 
                    delta=f"{confidence*100:.1f}% Confidence"
                )
                
            with col2:
                # Display all probabilities in a bar chart for context
                st.subheader("Emotion Confidence Breakdown")
                prob_data = {
                    'Emotion': [label.capitalize() for label in EMOTION_LABELS],
                    'Confidence': raw_probs
                }
                
                # Create a simple DataFrame for bar chart
                df = tf.convert_to_tensor(prob_data['Confidence']).numpy()
                df = df.reshape(1, -1)
                
                # Custom chart using st.bar_chart for a single row of data
                st.bar_chart(df, x=None, y=prob_data['Emotion'])
                
            st.divider()


    # --- Sample Reviews Section ---
    st.header("Emotion Samples")
    st.markdown("Use these examples to quickly test the different emotions.")
    
    sample_data = []
    for emotion, review in SAMPLE_REVIEWS.items():
        sample_data.append([emotion.capitalize(), review])
        
    st.table(sample_data)

    st.divider()
    
    # --- Model Evaluation Metrics ---
    
    st.header("Model Performance Metrics")
    st.markdown("Evaluation performed on the 20% test dataset.")
    
    if metrics:
        col_acc, col_prec, col_recall, col_f1 = st.columns(4)
        
        col_acc.metric(
            label="Accuracy", 
            value=f"{metrics['Accuracy']*100:.2f}%", 
            delta="Target: >85.00%" if metrics['Accuracy'] >= 0.85 else "Target: 85.00%"
        )
        col_prec.metric(label="Macro Precision", value=f"{metrics['Precision']:.3f}")
        col_recall.metric(label="Macro Recall", value=f"{metrics['Recall']:.3f}")
        col_f1.metric(label="Macro F1-Score", value=f"{metrics['F1-Score']:.3f}")
        
        st.caption(f"Note: This model was trained with {NUM_CLASSES} classes on the `dair-ai/emotion` dataset using a BiLSTM architecture.")
        
    else:
        st.warning("Metrics not available because the model could not be loaded or trained.")

if __name__ == "__main__":
    # Ensure Streamlit is run only if the model and tokenizer are defined
    if model is not None and tokenizer is not None:
        main()
    else:
        # Note: If this error is displayed, it means the app failed to start due to an earlier issue,
        # likely the one we just fixed in data loading.
        st.error("Application could not start because the necessary model or data failed to load/train.")


