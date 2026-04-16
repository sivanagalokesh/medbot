"""
Prediction Module with Confidence Score for Hybrid Chatbot
This module predicts intent and returns confidence for decision-making
"""
import numpy as np
from tensorflow.keras.models import load_model

from utils import load_intents, load_pickle, get_project_root, get_response
from preprocess import TextPreprocessor


class IntentPredictor:
    def __init__(self, model_path=None, tokenizer_path=None, label_encoder_path=None):
        """
        Initialize Intent Predictor with confidence tracking
        
        Args:
            model_path: Path to trained model
            tokenizer_path: Path to tokenizer pickle
            label_encoder_path: Path to label encoder pickle
        """
        project_root = get_project_root()
        models_dir = project_root / 'models'
        
        # Set default paths if not provided
        if model_path is None:
            model_path = models_dir / 'model.h5'
        if tokenizer_path is None:
            tokenizer_path = models_dir / 'tokenizer.pkl'
        if label_encoder_path is None:
            label_encoder_path = models_dir / 'label_encoder.pkl'
        
        # Load model and preprocessor
        self.model = load_model(model_path)
        self.tokenizer = load_pickle(tokenizer_path)
        self.label_encoder = load_pickle(label_encoder_path)
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor(max_words=1000, max_len=20)
        self.preprocessor.tokenizer = self.tokenizer
        self.preprocessor.label_encoder = self.label_encoder
        
        # Load intents for responses
        self.intents_data = load_intents()
        
        print("✓ LSTM Intent Model loaded successfully")
    
    def predict_intent_with_confidence(self, user_input):
        """
        Predict intent from user input with confidence score
        
        Args:
            user_input: User's text input
            
        Returns:
            tuple: (predicted_tag, confidence_score)
        """
        # Preprocess input
        sequence = self.preprocessor.texts_to_sequences([user_input])
        
        # Predict
        predictions = self.model.predict(sequence, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Decode prediction
        predicted_tag = self.label_encoder.inverse_transform([predicted_idx])[0]
        
        return predicted_tag, confidence
    
    def get_predefined_response(self, tag):
        """
        Get predefined safe response for a given tag
        
        Args:
            tag: Intent tag
            
        Returns:
            response: Predefined safe medical response
        """
        return get_response(tag, self.intents_data)
    
    def should_use_lstm_response(self, confidence, threshold=0.6):
        """
        Determine if LSTM prediction is confident enough
        
        Args:
            confidence: Confidence score from LSTM
            threshold: Minimum confidence threshold (default 0.6)
            
        Returns:
            bool: True if confidence >= threshold
        """
        return confidence >= threshold


def main():
    """Test prediction functionality"""
    print("=" * 70)
    print("Testing LSTM Intent Predictor with Confidence Scores")
    print("=" * 70)
    
    print("\nLoading predictor...")
    predictor = IntentPredictor()
    
    # Test predictions with various confidence levels
    test_inputs = [
        "I have a high fever",
        "My head is hurting badly",
        "I have a runny nose and sneezing",
        "Hello there",
        "What should I do about occasional dizziness?",  # Likely low confidence
        "I feel strange",  # Very low confidence
    ]
    
    print("\n" + "=" * 70)
    print("Test Predictions with Confidence")
    print("=" * 70)
    
    CONFIDENCE_THRESHOLD = 0.6
    
    for text in test_inputs:
        tag, confidence = predictor.predict_intent_with_confidence(text)
        
        print(f"\nInput: {text}")
        print(f"Predicted Intent: {tag}")
        print(f"Confidence: {confidence:.2%}")
        
        if predictor.should_use_lstm_response(confidence, CONFIDENCE_THRESHOLD):
            response = predictor.get_predefined_response(tag)
            print(f"✓ Using LSTM Response (High Confidence)")
            print(f"Response: {response}")
        else:
            print(f"✗ Low Confidence - Would use LLM fallback in hybrid system")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()