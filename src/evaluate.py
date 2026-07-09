"""Evaluation and visualization entry point."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .model import NeuralNetwork
    from .train import train_model
    from .utils import (
        FASHION_MNIST_LABELS,
        load_fashion_mnist,
        load_mnist,
        plot_confusion_matrix,
        plot_per_class_accuracy,
        plot_sample_predictions
    )
except ImportError:
    from model import NeuralNetwork
    from train import train_model
    from utils import (
        FASHION_MNIST_LABELS,
        load_fashion_mnist,
        load_mnist,
        plot_confusion_matrix,
        plot_per_class_accuracy,
        plot_sample_predictions
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = PROJECT_ROOT / 'plots'


def _load_dataset(dataset):
    if dataset == 'fashion_mnist':
        _, _, _, _, X_test, y_test = load_fashion_mnist()
        return X_test, y_test, FASHION_MNIST_LABELS
    if dataset == 'mnist':
        _, _, _, _, X_test, y_test = load_mnist()
        return X_test, y_test, [str(i) for i in range(10)]
    raise ValueError(f"Unknown dataset: {dataset}")


def evaluate_model(model_path, dataset='fashion_mnist', show_plots=True):
    print(f"Loading model from {model_path}")
    model = NeuralNetwork.load(model_path)

    print(f"Loading {dataset} dataset...")
    X_test, y_test, class_names = _load_dataset(dataset)

    print("Making predictions...")
    y_pred_logits = model.forward(X_test, training=False)
    test_loss = model.compute_loss(
        y_test,
        y_pred_logits,
        include_regularization=True
    )
    test_accuracy = model.compute_accuracy(y_test, y_pred_logits)

    y_test_np = y_test.numpy()
    y_pred_np = y_pred_logits.numpy()
    X_test_np = X_test.numpy()

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS - {dataset.upper()}")
    print("=" * 60)
    print(f"Test Loss: {float(test_loss):.4f}")
    print(f"Test Accuracy: {float(test_accuracy):.4f} ({float(test_accuracy) * 100:.2f}%)")
    print("=" * 60 + "\n")

    y_true_classes = np.argmax(y_test_np, axis=1)
    y_pred_classes = np.argmax(y_pred_np, axis=1)

    print("\nPer-Class Accuracy:")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        mask = y_true_classes == i
        if mask.sum() > 0:
            class_acc = (y_pred_classes[mask] == i).mean()
            print(f"{class_name:15s}: {class_acc * 100:6.2f}%")
    print("-" * 50)

    if show_plots:
        print("\nGenerating visualizations...")
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        plot_confusion_matrix(
            y_test_np,
            y_pred_np,
            class_names=class_names,
            title=f"Confusion Matrix - {dataset}",
            save_path=str(PLOTS_DIR / f"confusion_matrix_{dataset}.png")
        )
        plot_per_class_accuracy(y_test_np, y_pred_np, class_names)
        plot_sample_predictions(
            X_test_np[:100],
            y_test_np[:100],
            y_pred_np[:100],
            class_names,
            n_samples=16
        )

    return {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'predictions': y_pred_np,
        'true_labels': y_test_np
    }


def compare_loss_functions():
    print("=" * 60)
    print("COMPARING LOSS FUNCTIONS")
    print("=" * 60)

    base_config = {
        'num_layers': 3,
        'hidden_size': 128,
        'activation': 'relu',
        'weight_init': 'xavier',
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'weight_decay': 0.0005,
        'dataset': 'fashion_mnist'
    }

    print("\n1. Training with Cross-Entropy Loss...")
    config_ce = base_config.copy()
    config_ce['loss'] = 'cross_entropy'
    _, acc_ce, history_ce = train_model(
        config_ce,
        'loss_comparison',
        use_wandb=False
    )

    print("\n2. Training with MSE Loss...")
    config_mse = base_config.copy()
    config_mse['loss'] = 'mse'
    _, acc_mse, history_mse = train_model(
        config_mse,
        'loss_comparison',
        use_wandb=False
    )

    print("\n" + "=" * 60)
    print("LOSS FUNCTION COMPARISON RESULTS")
    print("=" * 60)
    print(f"Cross-Entropy Accuracy: {acc_ce:.4f} ({acc_ce * 100:.2f}%)")
    print(f"MSE Accuracy:          {acc_mse:.4f} ({acc_mse * 100:.2f}%)")
    print(f"Difference:            {(acc_ce - acc_mse) * 100:+.2f}%")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history_ce['val_loss']) + 1)

    axes[0].plot(epochs, history_ce['val_loss'], 'b-o', label='Cross-Entropy', linewidth=2)
    axes[0].plot(epochs, history_mse['val_loss'], 'r-s', label='MSE', linewidth=2)
    axes[0].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history_ce['val_accuracy'], 'b-o', label='Cross-Entropy', linewidth=2)
    axes[1].plot(epochs, history_mse['val_accuracy'], 'r-s', label='MSE', linewidth=2)
    axes[1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / 'loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_mnist_configs():
    configs = [
        {
            'name': 'Config 1: Conservative High-Performance',
            'num_layers': 3,
            'hidden_size': 128,
            'activation': 'relu',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'weight_decay': 0.0005,
            'batch_size': 32,
            'weight_init': 'xavier',
            'epochs': 10,
            'loss': 'cross_entropy',
            'dataset': 'mnist'
        },
        {
            'name': 'Config 2: Lightweight Efficient',
            'num_layers': 3,
            'hidden_size': 64,
            'activation': 'relu',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'weight_decay': 0.0,
            'batch_size': 64,
            'weight_init': 'xavier',
            'epochs': 10,
            'loss': 'cross_entropy',
            'dataset': 'mnist'
        },
        {
            'name': 'Config 3: Deep Exploration',
            'num_layers': 4,
            'hidden_size': 128,
            'activation': 'relu',
            'optimizer': 'nadam',
            'learning_rate': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 32,
            'weight_init': 'xavier',
            'epochs': 10,
            'loss': 'cross_entropy',
            'dataset': 'mnist'
        }
    ]

    results = []
    print("=" * 70)
    print("MNIST EVALUATION - 3 CONFIGURATIONS")
    print("=" * 70)

    for i, config in enumerate(configs, 1):
        print(f"\n{'=' * 70}")
        print(f"Configuration {i}: {config['name']}")
        print("=" * 70)

        config_copy = config.copy()
        name = config_copy.pop('name')
        _, test_acc, _ = train_model(
            config_copy,
            'mnist_evaluation',
            use_wandb=False
        )

        results.append({
            'config': name,
            'accuracy': test_acc
        })
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for result in results:
        print(f"{result['config']:45s}: {result['accuracy'] * 100:.2f}%")
    print("=" * 70)

    best = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest Configuration: {best['config']} ({best['accuracy'] * 100:.2f}%)")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to saved model without extension'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='fashion_mnist',
        choices=['fashion_mnist', 'mnist']
    )
    parser.add_argument('--no_plots', action='store_true')
    parser.add_argument('--compare_losses', action='store_true')
    parser.add_argument('--mnist_configs', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.compare_losses:
        compare_loss_functions()
    elif args.mnist_configs:
        evaluate_mnist_configs()
    elif args.model_path:
        evaluate_model(
            model_path=args.model_path,
            dataset=args.dataset,
            show_plots=not args.no_plots
        )
    else:
        print("Please specify --model_path, --compare_losses, or --mnist_configs")


if __name__ == "__main__":
    main()
