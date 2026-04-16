"""
LSTM Model Training Script for Medical Intent Classification
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from utils import load_intents, prepare_data_from_intents, save_pickle, get_project_root
from preprocess import TextPreprocessor


class IntentClassifier:
    def __init__(self, vocab_size, max_len, num_classes, embedding_dim=128, lstm_units=128):
        """
        Initialize Intent Classifier with LSTM
        
        Args:
            vocab_size: Size of vocabulary
            max_len: Maximum sequence length
            num_classes: Number of intent classes
            embedding_dim: Dimension of embedding layer
            lstm_units: Number of LSTM units
        """
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build LSTM model architecture"""
        model = Sequential([
            # Embedding layer
            Embedding(input_dim=self.vocab_size, 
                     output_dim=self.embedding_dim, 
                     input_length=self.max_len),
            
            # Bidirectional LSTM layer
            Bidirectional(LSTM(self.lstm_units, return_sequences=False)),
            
            # Dropout for regularization
            Dropout(0.5),
            
            # Dense hidden layer
            Dense(64, activation='relu'),
            Dropout(0.3),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=8):
        """
        Train the model
        
        Args:
            X_train: Training sequences
            y_train: Training labels (one-hot encoded)
            X_val: Validation sequences
            y_val: Validation labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        checkpoint_path = get_project_root() / 'models' / 'model.h5'
        model_checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, model_checkpoint],
            verbose=1
        )
        
        return self.history
    
    def save_model(self, filepath):
        """Save model to file"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def plot_training_history(self, save_path=None):
        """Plot training and validation accuracy/loss"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("HYBRID MEDICAL CHATBOT - LSTM Intent Classifier Training")
    print("=" * 70)
    
    # Load data
    print("\n[1/6] Loading intents data...")
    intents = load_intents()
    patterns, tags = prepare_data_from_intents(intents)
    print(f"✓ Loaded {len(patterns)} patterns across {len(set(tags))} intent classes")
    
    # Preprocess data
    print("\n[2/6] Preprocessing data...")
    preprocessor = TextPreprocessor(max_words=1000, max_len=20)
    preprocessor.fit_tokenizer(patterns)
    preprocessor.fit_label_encoder(tags)
    
    X = preprocessor.texts_to_sequences(patterns)
    y = preprocessor.encode_labels(tags)
    y_categorical = to_categorical(y)
    
    vocab_size = preprocessor.get_vocab_size()
    num_classes = preprocessor.get_num_classes()
    print(f"✓ Vocabulary size: {vocab_size}")
    print(f"✓ Number of classes: {num_classes}")
    
    # Split data
    print("\n[3/6] Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Build and train model
    print("\n[4/6] Building and training LSTM model...")
    classifier = IntentClassifier(
        vocab_size=vocab_size,
        max_len=preprocessor.max_len,
        num_classes=num_classes,
        embedding_dim=128,
        lstm_units=128
    )
    
    model = classifier.build_model()
    print(model.summary())
    
    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=8
    )
    
    # Save artifacts
    print("\n[5/6] Saving model and preprocessor...")
    project_root = get_project_root()
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    classifier.save_model(models_dir / 'model.h5')
    save_pickle(preprocessor.tokenizer, models_dir / 'tokenizer.pkl')
    save_pickle(preprocessor.label_encoder, models_dir / 'label_encoder.pkl')
    print("✓ All artifacts saved")
    
    # Plot and save training history
    print("\n[6/6] Generating training plots...")
    outputs_dir = project_root / 'outputs'
    outputs_dir.mkdir(exist_ok=True)
    classifier.plot_training_history(outputs_dir / 'accuracy_plot.png')
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final Train Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print("\n✓ Model ready for hybrid chatbot deployment!")
    print("=" * 70)


if __name__ == "__main__":
    main()