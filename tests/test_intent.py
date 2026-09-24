"""Unit and integration tests for AURA Phase 2 Intent Classification."""

import os
import json
import pytest
import torch

from app.brain.nlp import clean_and_tokenize, build_vocabulary, bag_of_words
from app.brain.dataset import IntentDataset, get_data_loader
from app.brain.model import IntentNeuralNet
from app.brain.train import prepare_training_data, train_intent_model
from app.brain.intent import IntentClassifier


# ==============================================================================
# 1. Dataset & Intents Schema Tests
# ==============================================================================

def test_intents_file_structure():
    """Verify data/intents.json contains required keys, intents, and non-empty patterns."""
    intents_path = "data/intents.json"
    assert os.path.exists(intents_path), f"{intents_path} does not exist"

    with open(intents_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "intents" in data
    assert len(data["intents"]) >= 8, "Expected at least 8 intents"

    tags = set()
    for item in data["intents"]:
        assert "tag" in item
        assert "patterns" in item
        assert "responses" in item
        assert len(item["patterns"]) > 0
        assert len(item["responses"]) > 0
        tags.add(item["tag"])

    # Core required intents
    expected_core_tags = {"greeting", "goodbye", "thanks", "time", "date", "weather", "calculator", "search"}
    assert expected_core_tags.issubset(tags)


# ==============================================================================
# 2. NLP Preprocessing Tests
# ==============================================================================

def test_clean_and_tokenize():
    """Verify text normalization and tokenization."""
    # Basic tokenization
    assert clean_and_tokenize("Hello World") == ["hello", "world"]
    
    # Punctuation stripping and lowercasing
    assert clean_and_tokenize("What's the weather like, Aura?!") == ["what", "s", "the", "weather", "like", "aura"]
    
    # Numbers and mixed characters
    assert clean_and_tokenize("Calculate 25 * 4 = 100") == ["calculate", "25", "4", "100"]
    
    # Empty string and whitespace
    assert clean_and_tokenize("") == []
    assert clean_and_tokenize("   \t\n  ") == []


def test_build_vocabulary():
    """Verify unique vocabulary creation and sorted ordering."""
    tokenized_sentences = [
        ["hello", "aura"],
        ["what", "is", "the", "time"],
        ["hello", "world"],
    ]
    vocab = build_vocabulary(tokenized_sentences)

    # All unique words
    assert set(vocab) == {"aura", "hello", "is", "the", "time", "what", "world"}
    # Must be sorted alphabetically
    assert vocab == sorted(vocab)
    assert len(vocab) == 7


# ==============================================================================
# 3. Feature Representation (Bag-of-Words) Tests
# ==============================================================================

def test_bag_of_words():
    """Verify Bag-of-Words numerical vector generation."""
    vocabulary = ["aura", "hello", "time", "weather"]
    
    # Sentence with known words
    bow1 = bag_of_words(["hello", "aura"], vocabulary)
    assert bow1 == [1.0, 1.0, 0.0, 0.0]
    assert len(bow1) == len(vocabulary)

    # Sentence with unseen words
    bow2 = bag_of_words(["goodbye", "mars"], vocabulary)
    assert bow2 == [0.0, 0.0, 0.0, 0.0]

    # Sentence with subset of words and punctuation already cleaned
    bow3 = bag_of_words(["time"], vocabulary)
    assert bow3 == [0.0, 0.0, 1.0, 0.0]


# ==============================================================================
# 4. PyTorch Dataset & DataLoader Tests
# ==============================================================================

def test_intent_dataset_and_dataloader():
    """Verify IntentDataset shapes, length, and DataLoader batching."""
    x_sample = [
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    y_sample = [0, 1, 0, 2]

    dataset = IntentDataset(x_sample, y_sample)
    assert len(dataset) == 4

    features, label = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert features.shape == (3,)
    assert label.item() == 0

    # Test DataLoader
    loader = get_data_loader(dataset, batch_size=2, shuffle=False)
    batch_count = 0
    for batch_x, batch_y in loader:
        assert batch_x.shape[0] <= 2
        assert batch_x.shape[1] == 3
        batch_count += 1
    assert batch_count == 2


# ==============================================================================
# 5. Neural Network Architecture Tests
# ==============================================================================

def test_model_architecture_and_forward_pass():
    """Verify IntentNeuralNet forward pass output shape and logits."""
    vocab_size = 50
    hidden_size = 16
    num_classes = 8
    batch_size = 4

    model = IntentNeuralNet(input_size=vocab_size, hidden_size=hidden_size, num_classes=num_classes)
    dummy_input = torch.randn(batch_size, vocab_size)

    output = model(dummy_input)
    assert output.shape == (batch_size, num_classes)
    # Output should be raw unnormalized logits
    assert isinstance(output, torch.Tensor)


# ==============================================================================
# 6. Training & Checkpoint Saving Tests
# ==============================================================================

def test_prepare_training_data():
    """Verify training data preparation from intents JSON."""
    x_train, y_train, vocab, tags, responses = prepare_training_data("data/intents.json")
    assert len(x_train) == len(y_train)
    assert len(x_train) > 0
    assert len(vocab) > 0
    assert len(tags) >= 8
    # Ensure every x sample matches the vocabulary dimension
    for sample in x_train:
        assert len(sample) == len(vocab)


def test_train_model_pipeline(tmp_path):
    """Verify training loop runs and saves valid model checkpoint and metadata."""
    model_path = str(tmp_path / "test_model.pth")
    meta_path = str(tmp_path / "test_meta.json")

    results = train_intent_model(
        intents_path="data/intents.json",
        model_save_path=model_path,
        metadata_save_path=meta_path,
        hidden_size=8,
        batch_size=16,
        learning_rate=0.01,
        num_epochs=10,  # Quick test run
    )

    assert os.path.exists(model_path)
    assert os.path.exists(meta_path)
    assert "final_loss" in results
    assert "final_accuracy" in results


# ==============================================================================
# 7. Inference & IntentClassifier Tests
# ==============================================================================

def test_intent_classifier_prediction():
    """Verify IntentClassifier produces accurate predictions and structured output."""
    classifier = IntentClassifier()

    # Common greetings
    result = classifier.predict("Hello Aura")
    assert isinstance(result, dict)
    assert "intent" in result
    assert "confidence" in result
    assert "response" in result
    assert result["intent"] == "greeting"
    assert result["confidence"] > 0.80

    # Weather query
    weather_res = classifier.predict("What is the weather today?")
    assert weather_res["intent"] == "weather"
    assert weather_res["confidence"] > 0.70

    # Time query
    time_res = classifier.predict("Tell me the time")
    assert time_res["intent"] == "time"
    assert time_res["confidence"] > 0.70

    # Calculator query
    calc_res = classifier.predict("Calculate 25 times 4")
    assert calc_res["intent"] == "calculator"
    assert calc_res["confidence"] > 0.70


def test_unknown_and_low_confidence_fallback():
    """Verify that unseen, nonsense, or low-confidence queries yield 'unknown'."""
    classifier = IntentClassifier(confidence_threshold=0.65)

    # 1. Out-of-vocabulary query (all words unseen)
    oov_result = classifier.predict("quantum xylophone zzyzx")
    assert oov_result["intent"] == "unknown"
    assert oov_result["confidence"] == 0.0

    # 2. Empty string or punctuation only
    empty_res = classifier.predict("")
    assert empty_res["intent"] == "unknown"
    assert empty_res["confidence"] == 0.0

    punct_res = classifier.predict("!?!?!?!")
    assert punct_res["intent"] == "unknown"
    assert punct_res["confidence"] == 0.0

    # 3. Explicitly trained unknown / gibberish pattern
    gibberish_res = classifier.predict("qwerty asdf gibberish")
    assert gibberish_res["intent"] == "unknown"


def test_custom_confidence_threshold():
    """Verify strict confidence threshold correctly flags lower probability queries."""
    strict_classifier = IntentClassifier(confidence_threshold=0.999999)
    # A slightly ambiguous or multi-word query won't hit 0.999999 confidence
    result = strict_classifier.predict("Tell me something cool")
    assert result["intent"] == "unknown"
