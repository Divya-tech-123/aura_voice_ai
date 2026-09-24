"""Training pipeline for AURA Intent Classifier.

This script:
1. Loads the intent dataset from data/intents.json.
2. Tokenizes patterns and builds the master vocabulary.
3. Encodes training sentences into Bag-of-Words vectors.
4. Initializes the IntentNeuralNet, CrossEntropyLoss, and Adam optimizer.
5. Trains the model across configurable epochs, logging loss and accuracy.
6. Saves model weights to models/intent_model.pth and metadata to models/intent_metadata.json.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.optim import Adam

from app.brain.nlp import clean_and_tokenize, build_vocabulary, bag_of_words
from app.brain.dataset import IntentDataset, get_data_loader
from app.brain.model import IntentNeuralNet

logger = logging.getLogger(__name__)


def prepare_training_data(
    intents_path: str,
) -> Tuple[list, list, list, list, Dict[str, list]]:
    """Load intents JSON and convert it into numerical features and labels.

    Args:
        intents_path: Path to intents.json.

    Returns:
        Tuple of (X_train, y_train, vocabulary, tags, tag_responses).
    """
    with open(intents_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_patterns = []
    xy_pairs = []
    tags = []
    tag_responses: Dict[str, list] = {}

    for intent in data["intents"]:
        tag = intent["tag"]
        if tag not in tags:
            tags.append(tag)
        tag_responses[tag] = intent.get("responses", [])

        for pattern in intent["patterns"]:
            tokens = clean_and_tokenize(pattern)
            all_patterns.append(tokens)
            xy_pairs.append((tokens, tag))

    # Deterministic alphabetical ordering for tags and vocabulary
    tags = sorted(list(set(tags)))
    vocabulary = build_vocabulary(all_patterns)

    # Map each tag to a distinct integer index (0 to num_classes - 1)
    tag_to_idx = {tag: i for i, tag in enumerate(tags)}

    x_train = []
    y_train = []
    for tokens, tag in xy_pairs:
        bow = bag_of_words(tokens, vocabulary)
        x_train.append(bow)
        y_train.append(tag_to_idx[tag])

    return x_train, y_train, vocabulary, tags, tag_responses


def train_intent_model(
    intents_path: str = "data/intents.json",
    model_save_path: str = "models/intent_model.pth",
    metadata_save_path: str = "models/intent_metadata.json",
    hidden_size: int = 16,
    batch_size: int = 8,
    learning_rate: float = 0.005,
    num_epochs: int = 200,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Train the intent classification neural network.

    Args:
        intents_path: Path to intents.json data file.
        model_save_path: Path where trained PyTorch model (.pth) will be saved.
        metadata_save_path: Path where vocabulary and tags JSON will be saved.
        hidden_size: Number of neurons in hidden layers.
        batch_size: Batch size for DataLoader.
        learning_rate: Learning rate for Adam optimizer.
        num_epochs: Total training epochs.
        device: 'cpu' or 'cuda'.

    Returns:
        Dictionary summarizing final training loss and accuracy.
    """
    print(f"Loading training data from '{intents_path}'...")
    x_train, y_train, vocabulary, tags, tag_responses = prepare_training_data(intents_path)

    input_size = len(vocabulary)
    num_classes = len(tags)
    print(f"Dataset prepared:")
    print(f"  Total samples:      {len(x_train)}")
    print(f"  Vocabulary size:    {input_size}")
    print(f"  Intent classes:     {num_classes} ({', '.join(tags)})")

    dataset = IntentDataset(x_train, y_train)
    train_loader = get_data_loader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model, Loss Function, and Optimizer
    model = IntentNeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    model.to(device)

    # CrossEntropyLoss combines LogSoftmax and NLLLoss in one single class
    criterion = nn.CrossEntropyLoss()
    # Adam (Adaptive Moment Estimation) dynamically adjusts learning rate per parameter
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print(f"\nStarting training for {num_epochs} epochs (lr={learning_rate}, batch_size={batch_size})...\n")

    final_loss = 0.0
    final_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for words, labels in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            # 1. Forward pass: compute predicted logits
            outputs = model(words)
            loss = criterion(outputs, labels)

            # 2. Backward pass: compute gradients
            optimizer.zero_grad()
            loss.backward()

            # 3. Optimization step: update weights
            optimizer.step()

            total_loss += loss.item() * words.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = total_loss / total
        epoch_acc = (correct / total) * 100.0
        final_loss = epoch_loss
        final_acc = epoch_acc

        if epoch == 1 or epoch % 25 == 0 or epoch == num_epochs:
            print(f"Epoch [{epoch:3d}/{num_epochs:3d}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:6.2f}%")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_save_path), exist_ok=True)

    # Save model checkpoint
    checkpoint = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": num_classes,
        "vocabulary": vocabulary,
        "tags": tags,
    }
    torch.save(checkpoint, model_save_path)
    print(f"\nTrained model weights saved to: {model_save_path}")

    # Save metadata JSON for quick inspection and inference mapping
    metadata = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": num_classes,
        "vocabulary": vocabulary,
        "tags": tags,
        "responses": tag_responses,
    }
    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata (vocabulary, tags, responses) saved to: {metadata_save_path}\n")

    return {
        "final_loss": final_loss,
        "final_accuracy": final_acc,
        "vocabulary_size": input_size,
        "classes": tags,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_intent_model()
