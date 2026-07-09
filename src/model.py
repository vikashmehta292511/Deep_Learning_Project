"""Model and optimizer helpers for Fashion-MNIST classification."""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf


class NeuralNetwork:
    """Feedforward neural network wrapper with an explicit GradientTape step."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: Optional[List[int]] = None,
        output_size: int = 10,
        activation: str = 'relu',
        weight_init: str = 'xavier',
        weight_decay: float = 0.0,
        random_seed: int = 42
    ):
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_name = activation
        self.weight_init = weight_init
        self.weight_decay = weight_decay
        self.model = self._build_model()
        self.config = {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'output_size': output_size,
            'activation': activation,
            'weight_init': weight_init,
            'weight_decay': weight_decay
        }

    def _get_activation(self):
        if self.activation_name == 'sigmoid':
            return tf.nn.sigmoid
        if self.activation_name == 'tanh':
            return tf.nn.tanh
        if self.activation_name == 'relu':
            return tf.nn.relu
        raise ValueError(f"Unknown activation: {self.activation_name}")

    def _get_initializer(self):
        if self.weight_init == 'xavier':
            return tf.keras.initializers.GlorotUniform()
        if self.weight_init == 'random':
            return tf.keras.initializers.RandomNormal(stddev=0.01)
        raise ValueError(f"Unknown weight initializer: {self.weight_init}")

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(self.input_size,)))

        activation = self._get_activation()
        for hidden_size in self.hidden_sizes:
            model.add(tf.keras.layers.Dense(
                hidden_size,
                activation=activation,
                kernel_initializer=self._get_initializer()
            ))

        model.add(tf.keras.layers.Dense(
            self.output_size,
            kernel_initializer=self._get_initializer()
        ))
        return model

    def forward(self, X, training: bool = False):
        return self.model(X, training=training)

    def regularization_loss(self):
        if self.weight_decay <= 0:
            return tf.constant(0.0, dtype=tf.float32)

        kernels = [
            variable for variable in self.model.trainable_variables
            if 'kernel' in variable.name
        ]
        if not kernels:
            return tf.constant(0.0, dtype=tf.float32)

        return self.weight_decay * tf.add_n(
            [tf.nn.l2_loss(variable) for variable in kernels]
        )

    def compute_loss(
        self,
        y_true,
        y_pred,
        loss_type='cross_entropy',
        include_regularization: bool = False
    ):
        if loss_type == 'cross_entropy':
            data_loss = tf.keras.losses.categorical_crossentropy(
                y_true,
                y_pred,
                from_logits=True
            )
        elif loss_type == 'mse':
            y_pred_probs = tf.nn.softmax(y_pred)
            data_loss = tf.reduce_sum(tf.square(y_true - y_pred_probs), axis=1)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        loss = tf.reduce_mean(data_loss)
        if include_regularization:
            loss += self.regularization_loss()
        return loss

    def compute_accuracy(self, y_true, y_pred):
        y_pred_classes = tf.argmax(y_pred, axis=1)
        y_true_classes = tf.argmax(y_true, axis=1)
        correct = tf.equal(y_pred_classes, y_true_classes)
        return tf.reduce_mean(tf.cast(correct, tf.float32))

    def train_step(self, X_batch, y_batch, optimizer, loss_type='cross_entropy'):
        with tf.GradientTape() as tape:
            predictions = self.forward(X_batch, training=True)
            loss = self.compute_loss(
                y_batch,
                predictions,
                loss_type,
                include_regularization=True
            )

        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        accuracy = self.compute_accuracy(y_batch, predictions)
        return loss, accuracy

    def predict(self, X):
        y_pred = self.forward(X, training=False)
        return tf.argmax(y_pred, axis=1)

    def save(self, filepath: str):
        base_path = Path(filepath)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        weights_path = base_path.with_name(base_path.name + '.weights.h5')
        self.model.save_weights(str(weights_path))

        config_path = base_path.with_name(base_path.name + '_config.json')
        with config_path.open('w', encoding='utf-8') as f:
            json.dump(self.config, f)

        print(f"Model saved to {base_path}")

    @classmethod
    def load(cls, filepath):
        base_path = Path(filepath)
        config_path = base_path.with_name(base_path.name + '_config.json')
        with config_path.open('r', encoding='utf-8') as f:
            config = json.load(f)

        model = cls(**config)
        weights_path = base_path.with_name(base_path.name + '.weights.h5')
        model.model.load_weights(str(weights_path))

        print(f"Model loaded from {base_path}")
        return model


class SGD:
    def __init__(self, learning_rate=0.01):
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    def apply_gradients(self, grads_and_vars):
        self.optimizer.apply_gradients(grads_and_vars)


class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum
        )

    def apply_gradients(self, grads_and_vars):
        self.optimizer.apply_gradients(grads_and_vars)


class Nesterov:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=True
        )

    def apply_gradients(self, grads_and_vars):
        self.optimizer.apply_gradients(grads_and_vars)


class RMSprop:
    def __init__(self, learning_rate=0.001, rho=0.9):
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            rho=rho
        )

    def apply_gradients(self, grads_and_vars):
        self.optimizer.apply_gradients(grads_and_vars)


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta1,
            beta_2=beta2
        )

    def apply_gradients(self, grads_and_vars):
        self.optimizer.apply_gradients(grads_and_vars)


class Nadam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.optimizer = tf.keras.optimizers.Nadam(
            learning_rate=learning_rate,
            beta_1=beta1,
            beta_2=beta2
        )

    def apply_gradients(self, grads_and_vars):
        self.optimizer.apply_gradients(grads_and_vars)


def get_optimizer(name, learning_rate=0.001, **kwargs):
    optimizers = {
        'sgd': SGD,
        'momentum': Momentum,
        'nag': Nesterov,
        'nesterov': Nesterov,
        'rmsprop': RMSprop,
        'adam': Adam,
        'nadam': Nadam
    }

    name = name.lower()
    if name not in optimizers:
        raise ValueError(
            f"Unknown optimizer: {name}. Choose from {list(optimizers.keys())}"
        )

    return optimizers[name](learning_rate=learning_rate, **kwargs)
