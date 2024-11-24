# model_utils.py
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def save_history_to_excel(history, base_path='drive/MyDrive/Colab Notebooks/flickr_logos_27_dataset/'):
    """Save the training history to an Excel file."""
    # Convert history data to a DataFrame
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    history_df['epoch'] = history_df['epoch'] + 1  # Optional: Make epoch 1-based index

    # Generate filename with timestamp
    dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"training_history_{dt_string}.xlsx"
    filepath = os.path.join(base_path, filename)

    # Save to Excel
    history_df.to_excel(filepath, index=False)
    print(f"Training history saved to {filepath}")

def plot_convergence_curves(history):
    """Plot training and validation accuracy and loss."""
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.legend()
    plt.legend(loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.legend()
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(true_labels, predictions, class_names):
    """
    Plot the confusion matrix for a set of labels and predictions.
    
    Parameters:
    - true_labels: array, true labels of the data.
    - predictions: array, model's predictions.
    - class_names: array, list of class names for the dataset.
    """
    true_labels = np.argmax(true_labels, axis=1)
    class_names = list(class_names)
    #print("True labels", true_labels)
    #print("predictions", predictions)
    #print("class names", class_names)
    #print(type(true_labels), type(predictions), type(class_names))
    # Compute the confusion matrix
  
    '''
    # Normalize the confusion matrix by row (i.e., by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names,
                cbar=True)  # Ensure colorbar is enabled
    
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.colorbar()
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()'''
        # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()