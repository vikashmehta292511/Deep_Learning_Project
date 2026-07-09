"""Training entry point for Fashion-MNIST and MNIST classification."""

import argparse
from pathlib import Path

import numpy as np
import wandb
from tqdm import tqdm

try:
    from .model import NeuralNetwork, get_optimizer
    from .utils import create_batches, load_fashion_mnist, load_mnist
except ImportError:
    from model import NeuralNetwork, get_optimizer
    from utils import create_batches, load_fashion_mnist, load_mnist


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / 'models'

DEFAULT_CONFIG = {
    'num_layers': 3,
    'hidden_size': 128,
    'activation': 'relu',
    'weight_init': 'xavier',
    'epochs': 10,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'weight_decay': 0.0,
    'loss': 'cross_entropy',
    'dataset': 'fashion_mnist'
}


def _normalize_config(config):
    normalized = DEFAULT_CONFIG.copy()
    normalized.update(dict(config))
    return normalized


def _run_name(config):
    return (
        f"hl_{config['num_layers']}_"
        f"hs_{config['hidden_size']}_"
        f"bs_{config['batch_size']}_"
        f"opt_{config['optimizer']}_"
        f"act_{config['activation']}"
    )


def _load_dataset(dataset_name):
    if dataset_name == 'mnist':
        return load_mnist()
    if dataset_name == 'fashion_mnist':
        return load_fashion_mnist()
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _evaluate_batches(model, dataset, loss_type):
    losses = []
    accuracies = []

    for X_batch, y_batch in dataset:
        y_pred = model.forward(X_batch, training=False)
        loss = model.compute_loss(
            y_batch,
            y_pred,
            loss_type,
            include_regularization=True
        )
        accuracy = model.compute_accuracy(y_batch, y_pred)
        losses.append(float(loss))
        accuracies.append(float(accuracy))

    return float(np.mean(losses)), float(np.mean(accuracies))


def train_model(
    config,
    project_name='dl_assignment',
    use_wandb=True,
    wandb_run=None
):
    """
    Train a neural network from a configuration dictionary.

    Args:
        config: Training hyperparameters.
        project_name: W&B project name when W&B is enabled.
        use_wandb: Enable W&B metric logging.
        wandb_run: Existing W&B run owned by a sweep agent.

    Returns:
        Tuple of model, test accuracy, and training history.
    """
    config = _normalize_config(config)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    run = wandb_run
    owns_wandb_run = False
    if use_wandb and run is None:
        run = wandb.init(
            project=project_name,
            config=config,
            name=_run_name(config)
        )
        owns_wandb_run = True
        config = _normalize_config(run.config)
    elif use_wandb and run is not None:
        run.name = run.name or _run_name(config)

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = _load_dataset(
        config['dataset']
    )

    print("Creating model...")
    hidden_sizes = [config['hidden_size']] * config['num_layers']
    model = NeuralNetwork(
        input_size=784,
        hidden_sizes=hidden_sizes,
        output_size=10,
        activation=config['activation'],
        weight_init=config['weight_init'],
        weight_decay=config.get('weight_decay', 0.0),
        random_seed=42
    )

    optimizer = get_optimizer(
        config['optimizer'],
        learning_rate=config['learning_rate']
    )

    print("Starting training...")
    best_val_accuracy = -np.inf
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        train_losses = []
        train_accuracies = []
        train_dataset = create_batches(
            X_train,
            y_train,
            config['batch_size'],
            shuffle=True
        )

        for X_batch, y_batch in tqdm(train_dataset, desc="Training"):
            loss, accuracy = model.train_step(
                X_batch,
                y_batch,
                optimizer,
                loss_type=config.get('loss', 'cross_entropy')
            )
            train_losses.append(float(loss))
            train_accuracies.append(float(accuracy))

        train_loss = float(np.mean(train_losses))
        train_accuracy = float(np.mean(train_accuracies))

        val_dataset = create_batches(
            X_val,
            y_val,
            config['batch_size'],
            shuffle=False
        )
        val_loss, val_accuracy = _evaluate_batches(
            model,
            val_dataset,
            config.get('loss', 'cross_entropy')
        )

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }
        if use_wandb and run is not None:
            run.log(metrics)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_name = (
                f"best_model_{run.id}"
                if use_wandb and run is not None
                else "best_model"
            )
            model.save(str(MODELS_DIR / model_name))

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    print("\nEvaluating on test set...")
    test_dataset = create_batches(
        X_test,
        y_test,
        config['batch_size'],
        shuffle=False
    )
    test_loss, test_accuracy = _evaluate_batches(
        model,
        test_dataset,
        config.get('loss', 'cross_entropy')
    )

    if use_wandb and run is not None:
        run.log({
            'test_accuracy': test_accuracy,
            'test_loss': test_loss
        })
        if owns_wandb_run:
            wandb.finish()

    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    return model, test_accuracy, history


def parse_args():
    parser = argparse.ArgumentParser(description='Train Neural Network')

    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument(
        '--activation',
        type=str,
        default='relu',
        choices=['sigmoid', 'tanh', 'relu']
    )
    parser.add_argument(
        '--weight_init',
        type=str,
        default='xavier',
        choices=['random', 'xavier']
    )
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['sgd', 'momentum', 'nag', 'nesterov', 'rmsprop', 'adam', 'nadam']
    )
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument(
        '--loss',
        type=str,
        default='cross_entropy',
        choices=['cross_entropy', 'mse']
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='fashion_mnist',
        choices=['fashion_mnist', 'mnist']
    )
    parser.add_argument('--project', type=str, default='dl_assignment')
    parser.add_argument('--no_wandb', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()
    config = {
        'num_layers': args.num_layers,
        'hidden_size': args.hidden_size,
        'activation': args.activation,
        'weight_init': args.weight_init,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'weight_decay': args.weight_decay,
        'loss': args.loss,
        'dataset': args.dataset
    }

    train_model(config, args.project, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
