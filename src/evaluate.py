"""
Model Evaluation Script - Generate Confusion Matrix and Metrics for PPT
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import load_model

from utils import load_intents, prepare_data_from_intents, load_pickle, get_project_root
from preprocess import TextPreprocessor


def load_trained_model():
    """Load trained model and preprocessor artifacts"""
    project_root = get_project_root()
    models_dir = project_root / 'models'
    
    model = load_model(models_dir / 'model.h5')
    tokenizer = load_pickle(models_dir / 'tokenizer.pkl')
    label_encoder = load_pickle(models_dir / 'label_encoder.pkl')
    
    return model, tokenizer, label_encoder


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate model performance
    
    Args:
        model: Trained Keras model
        X_test: Test sequences
        y_test: True labels (encoded)
        label_encoder: Label encoder for decoding
        
    Returns:
        predictions, accuracy, classification report
    """
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names)
    
    return y_pred, accuracy, report


def plot_confusion_matrix(y_true, y_pred, label_encoder, save_path=None):
    """
    Plot professional confusion matrix for PPT
    
    Args:
        y_true: True labels (encoded)
        y_pred: Predicted labels (encoded)
        label_encoder: Label encoder
        save_path: Path to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = label_encoder.classes_
    
    # Create figure with high DPI for PPT
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'},
                linewidths=0.5, linecolor='gray',
                annot_kws={'size': 12, 'weight': 'bold'})
    
    plt.title('Confusion Matrix - Hybrid Medical Chatbot\nLSTM Intent Classification', 
              fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('True Intent', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Intent', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_per_class_metrics(y_true, y_pred, label_encoder, save_path=None):
    """
    Plot per-class accuracy and F1 scores for PPT
    
    Args:
        y_true: True labels (encoded)
        y_pred: Predicted labels (encoded)
        label_encoder: Label encoder
        save_path: Path to save the plot
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    class_names = label_encoder.classes_
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names))
    )
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Per-class accuracy (based on diagonal of confusion matrix)
    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    bars1 = ax1.bar(class_names, per_class_accuracy, color='#4CAF50', edgecolor='darkgreen', linewidth=2)
    ax1.set_title('Per-Class Accuracy', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Intent Class', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1.1])
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 2: F1 Score
    bars2 = ax2.bar(class_names, f1, color='#2196F3', edgecolor='darkblue', linewidth=2)
    ax2.set_title('F1 Score per Intent', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlabel('Intent Class', fontsize=13, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1.1])
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Per-class metrics saved to {save_path}")
    
    plt.show()


def main():
    """Main evaluation pipeline"""
    print("=" * 70)
    print("HYBRID MEDICAL CHATBOT - Model Evaluation for Presentation")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading test data...")
    intents = load_intents()
    patterns, tags = prepare_data_from_intents(intents)
    print(f"✓ Loaded {len(patterns)} patterns")
    
    # Load model and preprocessor
    print("\n[2/5] Loading trained model...")
    model, tokenizer, label_encoder = load_trained_model()
    print("✓ Model and preprocessor loaded")
    
    # Preprocess data
    print("\n[3/5] Preprocessing data...")
    preprocessor = TextPreprocessor(max_words=1000, max_len=20)
    preprocessor.tokenizer = tokenizer
    preprocessor.label_encoder = label_encoder
    
    X = preprocessor.texts_to_sequences(patterns)
    y = preprocessor.encode_labels(tags)
    
    # Evaluate
    print("\n[4/5] Evaluating model...")
    y_pred, accuracy, report = evaluate_model(model, X, y, label_encoder)
    
    print(f"\n{'='*70}")
    print(f"OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*70}")
    print("\nDETAILED CLASSIFICATION REPORT:")
    print(report)
    
    # Generate visualizations for PPT
    print("\n[5/5] Generating visualizations for presentation...")
    project_root = get_project_root()
    outputs_dir = project_root / 'outputs'
    outputs_dir.mkdir(exist_ok=True)
    
    # Confusion Matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y, y_pred, label_encoder, 
                         outputs_dir / 'confusion_matrix.png')
    
    # Per-class metrics
    print("Generating per-class metrics...")
    plot_per_class_metrics(y, y_pred, label_encoder,
                          outputs_dir / 'per_class_metrics.png')
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print("\n✓ Confusion matrix saved for PPT")
    print("✓ Per-class metrics saved for PPT")
    print(f"\n Overall Accuracy: {accuracy*100:.2f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()