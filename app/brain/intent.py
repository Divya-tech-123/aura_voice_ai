"""Intent classification module for AURA.

Parses natural language user inputs and maps them to predefined system intents
using Bag-of-Words feature extraction and either a trained PyTorch Feed-Forward Neural Network
or a lightweight pattern-matching fallback when PyTorch is not installed.
"""

import os
import json
import random
import logging
from typing import Dict, Any, Optional, List, Set

from app.brain.nlp import clean_and_tokenize, bag_of_words

logger = logging.getLogger(__name__)

# Keyword hints to assist lightweight classification when PyTorch is unavailable
KEYWORD_TAG_HINTS: Dict[str, Set[str]] = {
    "greeting": {"hi", "hello", "hey", "greetings"},
    "goodbye": {"bye", "goodbye", "exit", "quit", "farewell"},
    "thanks": {"thanks", "thank", "appreciated", "grateful"},
    "time": {"time", "clock"},
    "date": {"date", "today", "day"},
    "weather": {"weather", "temperature", "forecast", "rain", "sunny", "climate"},
    "calculator": {"calculate", "compute", "math", "times", "divided", "plus", "minus", "sqrt", "equation"},
    "search": {"search", "google", "lookup", "online"},
    "capabilities": {"capabilities", "features", "commands", "features"},
}


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
        self.patterns_by_tag: Dict[str, List[str]] = {}
        self.model: Optional[Any] = None
        self._use_torch: bool = False

        self._load_or_train()

    def _load_or_train(self) -> None:
        """Load trained model and metadata, or trigger training if files do not exist."""
        # 1. Load raw intents definition if available to populate patterns and responses
        if os.path.exists(self.intents_path):
            try:
                with open(self.intents_path, "r", encoding="utf-8") as f:
                    intents_data = json.load(f)
                for item in intents_data.get("intents", []):
                    tag = item.get("tag")
                    if tag:
                        if tag not in self.tags:
                            self.tags.append(tag)
                        if tag not in self.responses:
                            self.responses[tag] = item.get("responses", [])
                        self.patterns_by_tag[tag] = item.get("patterns", [])
            except Exception as exc:
                logger.warning(f"Could not read raw intents from {self.intents_path}: {exc}")

        # 2. Check if PyTorch is installed and available
        torch_module = None
        try:
            import torch
            torch_module = torch
        except (ImportError, Exception):
            torch_module = None

        if torch_module is not None:
            # Auto-train if model or metadata missing
            if not os.path.exists(self.model_path) or not os.path.exists(self.metadata_path):
                if os.path.exists(self.intents_path):
                    logger.info("Trained model or metadata not found. Auto-training intent model...")
                    from app.brain.train import train_intent_model
                    train_intent_model(
                        intents_path=self.intents_path,
                        model_save_path=self.model_path,
                        metadata_save_path=self.metadata_path,
                    )

            if os.path.exists(self.metadata_path) and os.path.exists(self.model_path):
                try:
                    with open(self.metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    self.vocabulary = metadata["vocabulary"]
                    self.tags = metadata["tags"]
                    self.responses = metadata.get("responses", self.responses)
                    input_size = metadata["input_size"]
                    hidden_size = metadata["hidden_size"]
                    output_size = metadata["output_size"]

                    from app.brain.model import IntentNeuralNet
                    self.model = IntentNeuralNet(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_classes=output_size,
                    )

                    checkpoint = torch_module.load(self.model_path, map_location=self.device, weights_only=False)
                    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                        self.model.load_state_dict(checkpoint["model_state"])
                    else:
                        self.model.load_state_dict(checkpoint)

                    self.model.to(self.device)
                    self.model.eval()
                    self._use_torch = True
                    logger.info(f"IntentClassifier loaded PyTorch model ({output_size} classes, vocab={len(self.vocabulary)}).")
                    return
                except Exception as exc:
                    logger.warning(f"Failed to load PyTorch intent model: {exc}. Falling back to lightweight mode.")
                    self._use_torch = False
                    self.model = None

        # 3. Lightweight fallback mode (when PyTorch is not installed or model loading skipped)
        if os.path.exists(self.metadata_path) and not self.vocabulary:
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                self.vocabulary = metadata.get("vocabulary", [])
                if not self.tags:
                    self.tags = metadata.get("tags", [])
                if not self.responses:
                    self.responses = metadata.get("responses", {})
            except Exception:
                pass

        self._use_torch = False
        self.model = None
        logger.info(f"IntentClassifier initialized in lightweight non-torch mode ({len(self.tags)} tags).")

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict intent and confidence score for the given raw text query.

        Args:
            text: Raw user query string.

        Returns:
            Dict containing:
            - 'intent': Classified intent tag (str) or 'unknown'.
            - 'confidence': Softmax probability or similarity score (float between 0.0 and 1.0).
            - 'response': A contextual response string matching the intent.
        """
        # Step 1: Preprocessing & Tokenization
        tokens = clean_and_tokenize(text)

        # Edge case: empty or whitespace-only input
        if not tokens:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "response": self.get_random_response("unknown"),
            }

        # PyTorch inference path
        if self._use_torch and self.model is not None:
            return self._predict_torch(tokens, text)

        # Lightweight non-torch inference path
        return self._predict_lightweight(tokens, text)

    def _predict_torch(self, tokens: List[str], text: str) -> Dict[str, Any]:
        """Inference using the PyTorch neural network."""
        import torch

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

    def _predict_lightweight(self, tokens: List[str], text: str) -> Dict[str, Any]:
        """Lightweight deterministic intent prediction when PyTorch is not available."""
        # Check out-of-vocabulary if vocabulary is known
        if self.vocabulary:
            bow = bag_of_words(tokens, self.vocabulary)
            if sum(bow) == 0.0:
                return {
                    "intent": "unknown",
                    "confidence": 0.0,
                    "response": self.get_random_response("unknown"),
                }

        clean_text = text.lower().strip().strip("!?,.:;")
        clean_words = set(tokens)

        best_tag = "unknown"
        best_score = 0.0

        for tag, patterns in self.patterns_by_tag.items():
            if tag == "unknown":
                continue
            for pat in patterns:
                pat_clean = pat.lower().strip().strip("!?,.:;")
                pat_tokens = clean_and_tokenize(pat)
                if not pat_tokens:
                    continue

                if clean_text == pat_clean:
                    score = 0.98
                elif clean_text.startswith(pat_clean) or clean_text.endswith(pat_clean):
                    score = 0.90
                elif pat_clean in clean_text:
                    score = 0.85
                else:
                    pat_token_set = set(pat_tokens)
                    common = clean_words & pat_token_set
                    if common:
                        overlap_ratio = len(common) / len(pat_token_set)
                        score = 0.75 * overlap_ratio
                    else:
                        score = 0.0

                if score > best_score:
                    best_score = score
                    best_tag = tag

            # Apply keyword hint boost if applicable
            hints = KEYWORD_TAG_HINTS.get(tag, set())
            if hints and (clean_words & hints):
                hint_score = 0.85
                if hint_score > best_score:
                    best_score = hint_score
                    best_tag = tag

        # Check for unknown / gibberish patterns
        if "unknown" in self.patterns_by_tag:
            for unk_pat in self.patterns_by_tag["unknown"]:
                unk_clean = unk_pat.lower().strip()
                unk_tokens = set(clean_and_tokenize(unk_pat))
                if clean_text == unk_clean or (unk_tokens and (clean_words & unk_tokens)):
                    return {
                        "intent": "unknown",
                        "confidence": 0.0,
                        "response": self.get_random_response("unknown"),
                    }

        if best_score < self.confidence_threshold:
            return {
                "intent": "unknown",
                "confidence": round(best_score, 4),
                "response": self.get_random_response("unknown"),
            }

        return {
            "intent": best_tag,
            "confidence": round(best_score, 4),
            "response": self.get_random_response(best_tag),
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
