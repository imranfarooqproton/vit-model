"""
Model utilities for Insulator Classification App
Handles model loading, image preprocessing, and predictions
"""

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTImageProcessor
import os
import json
from typing import Tuple, Dict
import config


class InsulatorClassifier:
    """
    Wrapper class for insulator damage classification using ViT model
    """
    
    def __init__(self):
        """Initialize the classifier"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_extractor = None
        self.transform = None
        self.idx_to_class = None  # Loaded from class_mapping.json
        self.class_names = None   # Ordered list of class names
        self._setup_transform()
        
    def _setup_transform(self):
        """Setup image transformation pipeline"""
        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ])

    def _load_class_mapping(self, model_path: str):
        """
        Load class mapping from class_mapping.json saved during training.
        Falls back to config.CLASS_NAMES if no mapping file exists.
        """
        mapping_path = os.path.join(model_path, "class_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
            # idx_to_class keys are strings in JSON; convert to int
            self.idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
            self.class_names = mapping.get("classes", config.CLASS_NAMES)
            print(f"[OK] Class mapping loaded: {self.idx_to_class}")
        else:
            # Fallback: assume alphabetical order (same as ImageFolder default)
            self.idx_to_class = {i: name for i, name in enumerate(config.CLASS_NAMES)}
            self.class_names = list(config.CLASS_NAMES)
            print("[WARNING] No class_mapping.json found, using config.CLASS_NAMES order")

    def is_trained_model(self, model_path: str = None) -> bool:
        """Check if a trained model with class mapping exists at the given path."""
        print("Checking for trained model...") # Debug print
        if model_path is None:
            model_path = config.MODEL_PATH
        mapping_path = os.path.join(model_path, "class_mapping.json")
        safetensors_path = os.path.join(model_path, "model.safetensors")
        exists = os.path.exists(mapping_path) and (os.path.exists(safetensors_path) or os.path.exists(os.path.join(model_path, "pytorch_model.bin")))
        print(f"Model exists at {model_path}: {exists}")
        return exists
    
    def load_model(self, model_path: str = None) -> bool:
        """
        Load the ViT model and feature extractor
        
        Args:
            model_path: Path to saved model. If None, uses config.MODEL_PATH
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if model_path is None:
            model_path = config.MODEL_PATH
            
        try:
            # Try loading local model first
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}...")
                self.model = ViTForImageClassification.from_pretrained(model_path)
                self.feature_extractor = ViTImageProcessor.from_pretrained(model_path)
                self._load_class_mapping(model_path)
                print("[OK] Local model loaded successfully!")
            else:
                # Fallback to pretrained model
                print(f"Local model not found at {model_path}")
                print(f"Loading pretrained model: {config.FALLBACK_MODEL}...")
                self.model = ViTForImageClassification.from_pretrained(
                    config.FALLBACK_MODEL,
                    num_labels=len(config.CLASS_NAMES),
                    ignore_mismatched_sizes=True
                )
                self.feature_extractor = ViTImageProcessor.from_pretrained(config.FALLBACK_MODEL)
                self.idx_to_class = {i: name for i, name in enumerate(config.CLASS_NAMES)}
                self.class_names = list(config.CLASS_NAMES)
                print("[WARNING] Using pretrained model (not fine-tuned on your dataset)")
                
            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading model: {str(e)}")
            return False
    
    def predict(self, image: Image.Image) -> Dict:
        """
        Predict class and confidence for an input image
        
        Args:
            image: PIL Image object
            
        Returns:
            dict: Contains 'class', 'confidence', 'all_probabilities'
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess image using feature extractor
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                confidence, predicted_class_id = torch.max(probs, dim=1)
            
            # Use class mapping from training (not hard-coded config)
            predicted_class = self.idx_to_class[predicted_class_id.item()]
            confidence_score = confidence.item()
            
            # Get all class probabilities using correct mapping
            all_probs = {
                self.idx_to_class[i]: probs[0][i].item() 
                for i in range(len(self.class_names))
            }
            
            return {
                'class': predicted_class,
                'confidence': confidence_score,
                'all_probabilities': all_probs,
                'class_id': predicted_class_id.item()
            }
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    def get_device_info(self) -> str:
        """Get information about the device being used"""
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}"
        else:
            return "CPU"
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for model input
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    return transform(image)


def get_confidence_color(confidence: float) -> str:
    """
    Get color based on confidence level
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Hex color code
    """
    if confidence >= 0.9:
        return "#00C853"  # Green
    elif confidence >= 0.75:
        return "#FFC107"  # Amber
    else:
        return "#FF5252"  # Red


def format_confidence(confidence: float) -> str:
    """
    Format confidence score as percentage
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Formatted string (e.g., "95.2%")
    """
    return f"{confidence * 100:.1f}%"
