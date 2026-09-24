"""PyTorch Feed-Forward Neural Network for AURA Intent Classification."""

import torch
import torch.nn as nn


class IntentNeuralNet(nn.Module):
    """A 3-layer Feed-Forward Neural Network (Multilayer Perceptron) for Intent Classification.

    Architecture:
        Input Layer  (size = vocab_size)
             ↓
        Linear Layer (vocab_size -> hidden_size)
             ↓
        ReLU Activation
             ↓
        Linear Layer (hidden_size -> hidden_size)
             ↓
        ReLU Activation
             ↓
        Output Layer (hidden_size -> num_classes)
             ↓
        Raw Logits   (shape: batch_size, num_classes)

    Note:
        We output unnormalized raw logits. PyTorch's `nn.CrossEntropyLoss` automatically
        applies `LogSoftmax` followed by `NLLLoss` internally for numerical stability.
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        """Initialize neural network layers.

        Args:
            input_size: Dimension of input vector (length of vocabulary).
            hidden_size: Number of neurons in hidden layers (e.g., 16 or 32).
            num_classes: Number of intent classes to predict (e.g., 10).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Layer 1: Input to first hidden layer
        self.l1 = nn.Linear(input_size, hidden_size)
        # Layer 2: First hidden layer to second hidden layer
        self.l2 = nn.Linear(hidden_size, hidden_size)
        # Layer 3: Second hidden layer to output layer (logits)
        self.l3 = nn.Linear(hidden_size, num_classes)
        # Activation function introducing non-linearity: ReLU(x) = max(0, x)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Args:
            x: Input batch tensor of shape (batch_size, input_size).

        Returns:
            Output logits tensor of shape (batch_size, num_classes).
        """
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # No softmax here because CrossEntropyLoss applies it internally during training
        return out
