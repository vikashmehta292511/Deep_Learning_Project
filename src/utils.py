"""
Contains functions for:
- Loading and preprocessing data
- Creating batches
- Plotting results
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_fashion_mnist(validation_split=0.1):
    """
    Load Fashion-MNIST dataset and prepare it for training
    
    Fashion-MNIST: 70,000 grayscale images of clothing items
    - 60,000 for training (we'll split this into train + validation)
    - 10,000 for testing
    - 10 classes (T-shirt, Trouser, Pullover, etc.)
    
    Returns: X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Load data from Keras datasets
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Normalize to [0, 1]
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Flatten images from 28x28 to 784
    X_train_full = X_train_full.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
    
    # Split into train and validation
    n_train = int(len(X_train_full) * (1 - validation_split))
    
    X_train = X_train_full[:n_train]
    y_train = y_train_full[:n_train]
    X_val = X_train_full[n_train:]
    y_val = y_train_full[n_train:]
    
    # Convert to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Convert to TensorFlow tensors
    X_train = tf.constant(X_train, dtype=tf.float32)
    y_train = tf.constant(y_train, dtype=tf.float32)
    X_val = tf.constant(X_val, dtype=tf.float32)
    y_val = tf.constant(y_val, dtype=tf.float32)
    X_test = tf.constant(X_test, dtype=tf.float32)
    y_test = tf.constant(y_test, dtype=tf.float32)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_mnist(validation_split=0.1):
    """
    Load and preprocess MNIST dataset
    
    Args:
        validation_split: Fraction of training data for validation
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Load data
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize to [0, 1]
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Flatten images from 28x28 to 784
    X_train_full = X_train_full.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
    
    # Split into train and validation
    n_train = int(len(X_train_full) * (1 - validation_split))
    
    X_train = X_train_full[:n_train]
    y_train = y_train_full[:n_train]
    X_val = X_train_full[n_train:]
    y_val = y_train_full[n_train:]
    
    # Convert to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Convert to TensorFlow tensors
    X_train = tf.constant(X_train, dtype=tf.float32)
    y_train = tf.constant(y_train, dtype=tf.float32)
    X_val = tf.constant(X_val, dtype=tf.float32)
    y_val = tf.constant(y_val, dtype=tf.float32)
    X_test = tf.constant(X_test, dtype=tf.float32)
    y_test = tf.constant(y_test, dtype=tf.float32)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_batches(X, y, batch_size, shuffle=True):
    """
    Create batches from dataset
    
    Args:
        X: Input data
        y: Labels
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Yields:
        Batches of (X_batch, y_batch)
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    dataset = dataset.batch(batch_size)
    
    return dataset


# Fashion-MNIST class labels
FASHION_MNIST_LABELS = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    title='Confusion Matrix',
    figsize=(12, 10),
    save_path=None
):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels (one-hot or indices)
        y_pred: Predicted labels (one-hot or indices)
        class_names: List of class names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    # Convert to class indices if one-hot
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot counts
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1,
        cbar_kws={'label': 'Count'}
    )
    ax1.set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Plot normalized
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax2,
        cbar_kws={'label': 'Proportion'}
    )
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return cm


def plot_training_history(history, figsize=(14, 5)):
    """
    Plot training history
    
    Args:
        history: Dictionary with training metrics
        figsize: Figure size
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_accuracy'], 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracy'], 'r-s', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_per_class_accuracy(y_true, y_pred, class_names, figsize=(12, 6)):
    """
    Plot per-class accuracy
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
    """
    # Convert to class indices
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Compute per-class accuracy
    accuracies = []
    for i in range(len(class_names)):
        mask = y_true == i
        if mask.sum() > 0:
            accuracy = (y_pred[mask] == i).mean()
            accuracies.append(accuracy * 100)
        else:
            accuracies.append(0)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(range(len(class_names)), accuracies, color='steelblue', alpha=0.8)
    
    # Color based on performance
    for i, bar in enumerate(bars):
        if accuracies[i] >= 90:
            bar.set_color('green')
        elif accuracies[i] >= 80:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_sample_predictions(X, y_true, y_pred, class_names, n_samples=16, figsize=(12, 12)):
    """
    Plot sample predictions
    
    Args:
        X: Input images
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        n_samples: Number of samples to plot
        figsize: Figure size
    """
    # Convert to numpy if tensor
    if isinstance(X, tf.Tensor):
        X = X.numpy()
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()
    
    # Convert to class indices
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Reshape images
    X_images = X.reshape(-1, 28, 28)
    
    # Select random samples
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    
    rows = int(np.sqrt(n_samples))
    cols = int(np.ceil(n_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, ax in zip(indices, axes):
        ax.imshow(X_images[idx], cmap='gray')
        
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', 
                     color=color, fontsize=10)
        ax.axis('off')
    
    # Hide extra subplots
    for idx in range(len(indices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
