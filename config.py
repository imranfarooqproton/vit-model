"""
Configuration file for Insulator Classification App
Contains model settings, paths, and UI styling constants
"""

import os

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model paths - update these based on your setup
MODEL_PATH = "./vit_model"  # Path to saved ViT model
FALLBACK_MODEL = "google/vit-base-patch16-224"  # Pretrained model if local not found

# Dataset paths
TRAIN_DATA_PATH = "./dataset/train/"
TEST_DATA_PATH = "./dataset/test/"

# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# Class labels
CLASS_NAMES = ["damage", "normal"]
CLASS_COLORS = {
    "damage": "#FF4B4B",  # Red for damage
    "normal": "#00C853"   # Green for normal
}
CLASS_ICONS = {
    "damage": "⚠️",
    "normal": "✅"
}

# Image preprocessing settings
IMAGE_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ============================================================================
# APP CONFIGURATION
# ============================================================================

APP_TITLE = "🔌 Insulator Health Monitor"
APP_SUBTITLE = "AI-Powered Insulator Damage Detection"
APP_DESCRIPTION = """
Advanced Vision Transformer (ViT) model for real-time detection of insulator damage.
Upload an image to get instant classification with confidence scores.
"""

# Supported image formats
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp", "tiff"]

# Confidence threshold for warnings
CONFIDENCE_THRESHOLD = 0.75

# ============================================================================
# UI STYLING
# ============================================================================

# Color scheme
PRIMARY_COLOR = "#1E88E5"
SECONDARY_COLOR = "#00ACC1"
BACKGROUND_GRADIENT = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
CARD_BACKGROUND = "rgba(255, 255, 255, 0.95)"

# Custom CSS
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    /* Card styling */
    .result-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    
    /* Upload section */
    .upload-section {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 2rem;
        border: 2px dashed #667eea;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #764ba2;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    /* Prediction result */
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .damage {
        background: linear-gradient(135deg, #FF6B6B 0%, #EE5A6F 100%);
        color: white;
    }
    
    .normal {
        background: linear-gradient(135deg, #51CF66 0%, #37B24D 100%);
        color: white;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Confidence meter */
    .confidence-meter {
        background: #f0f0f0;
        border-radius: 10px;
        height: 30px;
        position: relative;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
"""

# ============================================================================
# MODEL INFO
# ============================================================================

MODEL_INFO = {
    "Architecture": "Vision Transformer (ViT)",
    "Base Model": "google/vit-base-patch16-224",
    "Input Size": "224x224 pixels",
    "Classes": 2,
    "Framework": "PyTorch + Hugging Face Transformers"
}
