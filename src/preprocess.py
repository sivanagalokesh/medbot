"""
Text preprocessing module for hybrid medical chatbot
"""
import re
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np


class TextPreprocessor:
    def __init__(self, max_words=1000, max_len=20):
        """
        Initialize text preprocessor
        
        Args:
            max_words: Maximum number of words in vocabulary
            max_len: Maximum sequence length for padding
        """
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.label_encoder = LabelEncoder()
    
    def clean_text(self, text):
        """
        Clean and normalize text
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def fit_tokenizer(self, texts):
        """
        Fit tokenizer on texts
        
        Args:
            texts: List of text strings
        """
        cleaned_texts = [self.clean_text(text) for text in texts]
        self.tokenizer.fit_on_texts(cleaned_texts)
        
    def texts_to_sequences(self, texts, padding=True):
        """
        Convert texts to padded sequences
        
        Args:
            texts: List of text strings
            padding: Whether to pad sequences
            
        Returns:
            Numpy array of sequences
        """
        cleaned_texts = [self.clean_text(text) for text in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        
        if padding:
            sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        
        return np.array(sequences)
    
    def fit_label_encoder(self, labels):
        """
        Fit label encoder on tags
        
        Args:
            labels: List of label strings
        """
        self.label_encoder.fit(labels)
    
    def encode_labels(self, labels):
        """
        Encode labels to integers
        
        Args:
            labels: List of label strings
            
        Returns:
            Numpy array of encoded labels
        """
        return self.label_encoder.transform(labels)
    
    def decode_labels(self, encoded_labels):
        """
        Decode integers to labels
        
        Args:
            encoded_labels: Array of encoded labels
            
        Returns:
            List of label strings
        """
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def get_vocab_size(self):
        """Get vocabulary size"""
        return min(len(self.tokenizer.word_index) + 1, self.max_words)
    
    def get_num_classes(self):
        """Get number of classes"""
        return len(self.label_encoder.classes_)


if __name__ == "__main__":
    # Test preprocessing
    from utils import load_intents, prepare_data_from_intents
    
    intents = load_intents()
    patterns, tags = prepare_data_from_intents(intents)
    
    preprocessor = TextPreprocessor(max_words=1000, max_len=20)
    
    # Fit and transform
    preprocessor.fit_tokenizer(patterns)
    preprocessor.fit_label_encoder(tags)
    
    sequences = preprocessor.texts_to_sequences(patterns)
    encoded_labels = preprocessor.encode_labels(tags)
    
    print(f"Vocabulary size: {preprocessor.get_vocab_size()}")
    print(f"Number of classes: {preprocessor.get_num_classes()}")
    print(f"Sample sequence shape: {sequences.shape}")
    print(f"Sample encoded labels: {encoded_labels[:5]}")