"""Intent classification module for AURA.

Parses natural language user inputs and maps them to predefined system intents
using Bag-of-Words feature extraction and a trained PyTorch Feed-Forward Neural Network.
"""

import os
import json
import random
import logging
from typing import Dict, Any, Optional, List
import torch

from app.brain.nlp import clean_and_tokenize, bag_of_words
from app.brain.model import IntentNeuralNet

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classifies user queries into actionable intents with confidence scores."""

    DEFAULT_MODEL_PATH = "models/intent_model.pth"
    DEFAULT_METADATA_PATH = "models/intent_metadata.json"
    DEFAULT_INTENTS_PATH = "data/intents.json"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.65

    def __init__(
        self,
        model_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        intents_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: str = "cpu",
    ) -> None:
        """Initialize the intent classifier and load model weights and vocabulary.

        Args:
            model_path: Path to trained PyTorch weights (.pth). Defaults to models/intent_model.pth.
            metadata_path: Path to metadata JSON. Defaults to models/intent_metadata.json.
            intents_path: Path to raw intents JSON. Defaults to data/intents.json.
            confidence_threshold: Minimum softmax probability required to accept an intent.
            device: Computing device ('cpu' or 'cuda').
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.metadata_path = metadata_path or self.DEFAULT_METADATA_PATH
        self.intents_path = intents_path or self.DEFAULT_INTENTS_PATH
        self.confidence_threshold = confidence_threshold
        self.device = device

        self.vocabulary: List[str] = []
        self.tags: List[str] = []
        self.responses: Dict[str, List[str]] = {}
        self.model: Optional[IntentNeuralNet] = None

        self._load_or_train()

    def _load_or_train(self) -> None:
        """Load trained model and metadata, or trigger training if files do not exist."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.metadata_path):
            logger.info("Trained model or metadata not found. Auto-training intent model...")
            from app.brain.train import train_intent_model
            train_intent_model(
                intents_path=self.intents_path,
                model_save_path=self.model_path,
                metadata_save_path=self.metadata_path,
            )

        # Load metadata
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.vocabulary = metadata["vocabulary"]
        self.tags = metadata["tags"]
        self.responses = metadata.get("responses", {})
        input_size = metadata["input_size"]
        hidden_size = metadata["hidden_size"]
        output_size = metadata["output_size"]

        # Instantiate model architecture
        self.model = IntentNeuralNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=output_size,
        )

        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"IntentClassifier loaded successfully ({output_size} classes, vocab={len(self.vocabulary)}).")

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict intent and confidence score for the given raw text query.

        Pipeline:
        1. Tokenize & clean user input.
        2. Convert tokens into Bag-of-Words representation.
        3. Forward pass through neural network.
        4. Apply softmax to get probability distribution.
        5. Verify top probability against confidence threshold.

        Args:
            text: Raw user query string.

        Returns:
            Dict containing:
            - 'intent': Classified intent tag (str) or 'unknown'.
            - 'confidence': Softmax probability score (float between 0.0 and 1.0).
            - 'response': A contextual response string matching the intent.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        # Step 1: Preprocessing & Tokenization
        tokens = clean_and_tokenize(text)

        # Edge case: empty or whitespace-only input
        if not tokens:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "response": self.get_random_response("unknown"),
            }

        # Step 2: Numerical Feature Vector (Bag of Words)
        bow_vector = bag_of_words(tokens, self.vocabulary)

        # Check if any words in the query match our vocabulary
        if sum(bow_vector) == 0.0:
            # Completely out-of-vocabulary query
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "response": self.get_random_response("unknown"),
            }

        # Convert to PyTorch tensor of shape (1, vocab_size)
        x_tensor = torch.tensor([bow_vector], dtype=torch.float32).to(self.device)

        # Step 3: Forward Pass
        with torch.no_grad():
            logits = self.model(x_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence_tensor, predicted_idx_tensor = torch.max(probabilities, dim=1)

            confidence = float(confidence_tensor.item())
            predicted_idx = int(predicted_idx_tensor.item())
            predicted_tag = self.tags[predicted_idx]

        # Step 4: Unknown Intent Threshold Gate
        if confidence < self.confidence_threshold:
            return {
                "intent": "unknown",
                "confidence": round(confidence, 4),
                "response": self.get_random_response("unknown"),
            }

        return {
            "intent": predicted_tag,
            "confidence": round(confidence, 4),
            "response": self.get_random_response(predicted_tag),
        }

    def get_random_response(self, intent_tag: str) -> str:
        """Fetch a random canned response for a given intent tag.

        Args:
            intent_tag: Intent name.

        Returns:
            A response string.
        """
        responses = self.responses.get(intent_tag)
        if responses:
            return random.choice(responses)
        return "I am processing your request."
