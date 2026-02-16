"""
Professional Streamlit App for Insulator Damage Classification
Using Vision Transformer (ViT) Model
"""

import streamlit as st
from PIL import Image
import io
import time
from pathlib import Path
import config
from model_utils import InsulatorClassifier, format_confidence, get_confidence_color

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'classifier' not in st.session_state:
    st.session_state.classifier = None
    st.session_state.model_loaded = False
    st.session_state.training_done = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_classifier():
    """Load and cache the classifier model"""
    classifier = InsulatorClassifier()
    success = classifier.load_model()
    return classifier, success


def auto_train_model():
    """
    Auto-train the model with Streamlit progress UI.
    Returns True if training succeeded.
    """
    from train_model import train_model

    st.markdown("""
        <div class="result-card">
            <h3>🏋️ Training AI Model</h3>
            <p>No trained model found. Auto-training on your dataset...</p>
            <p style="font-size: 0.85rem; color: #666;">
                This may take 10-30 minutes depending on your hardware.
            </p>
        </div>
    """, unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    log_area = st.empty()
    log_lines = []

    def progress_callback(progress=None, msg=None):
        if msg:
            log_lines.append(msg)
            status_text.markdown(f"**Status:** {msg}")
            # Show last 8 log lines
            log_area.code("\n".join(log_lines[-8:]), language="text")
        if progress is not None:
            progress_bar.progress(min(progress, 100))

    result = train_model(progress_callback=progress_callback)

    progress_bar.empty()
    status_text.empty()
    log_area.empty()

    if result["success"]:
        best_acc = result.get("best_accuracy", 0)
        per_class = result.get("per_class_accuracy", {})
        st.success(f"✅ Training complete! Best accuracy: **{best_acc:.1f}%**")
        for cls, acc in per_class.items():
            st.write(f"  - **{cls}**: {acc:.1f}%")
        return True
    else:
        st.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")
        return False


def display_header():
    """Display the app header"""
    st.markdown("""
        <div class="app-header fade-in">
            <h1>🔌 Insulator Health Monitor</h1>
            <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                AI-Powered Insulator Damage Detection System
            </p>
            <p style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">
                Powered by Vision Transformer (ViT) Deep Learning Model
            </p>
        </div>
    """, unsafe_allow_html=True)


def display_prediction_result(result: dict, image: Image.Image):
    """
    Display prediction results with beautiful formatting
    
    Args:
        result: Prediction result dictionary
        image: Original image
    """
    predicted_class = result['class']
    confidence = result['confidence']
    all_probs = result['all_probabilities']
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Uploaded Image")
        st.image(image, use_container_width=True, caption="Input Image")
    
    with col2:
        st.markdown("### 🎯 Prediction Results")
        
        # Display prediction with icon and color
        icon = config.CLASS_ICONS[predicted_class]
        color = config.CLASS_COLORS[predicted_class]
        
        st.markdown(f"""
            <div class="prediction-result {predicted_class} fade-in">
                {icon} {predicted_class.upper()}
            </div>
        """, unsafe_allow_html=True)
        
        # Confidence score
        st.markdown(f"**Confidence:** {format_confidence(confidence)}")
        
        # Confidence bar
        conf_color = get_confidence_color(confidence)
        st.markdown(f"""
            <div class="confidence-meter">
                <div class="confidence-fill" style="width: {confidence*100}%; background: {conf_color};"></div>
            </div>
        """, unsafe_allow_html=True)
        
        # Warning if low confidence
        if confidence < config.CONFIDENCE_THRESHOLD:
            st.warning(f"⚠️ Low confidence ({format_confidence(confidence)}). Consider manual inspection.")
        else:
            st.success(f"✓ High confidence prediction ({format_confidence(confidence)})")
    
    # Detailed probabilities
    st.markdown("---")
    st.markdown("### 📊 Class Probabilities")
    
    prob_cols = st.columns(len(config.CLASS_NAMES))
    for idx, (class_name, prob) in enumerate(all_probs.items()):
        with prob_cols[idx]:
            icon = config.CLASS_ICONS[class_name]
            color = config.CLASS_COLORS[class_name]
            st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: {color}20; 
                     border-radius: 10px; border: 2px solid {color};">
                    <div style="font-size: 2rem;">{icon}</div>
                    <div style="font-weight: bold; margin: 0.5rem 0;">{class_name.upper()}</div>
                    <div style="font-size: 1.5rem; color: {color};">
                        {format_confidence(prob)}
                    </div>
                </div>
            """, unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar with app information"""
    with st.sidebar:
        st.markdown("## 📋 About")
        st.markdown(config.APP_DESCRIPTION)
        
        st.markdown("---")
        st.markdown("## ⚙️ Model Information")
        
        for key, value in config.MODEL_INFO.items():
            st.markdown(f"**{key}:** {value}")
        
        # Device info
        if st.session_state.classifier is not None:
            device = st.session_state.classifier.get_device_info()
            st.markdown(f"**Device:** {device}")
        
        st.markdown("---")
        st.markdown("## 📊 Classification Classes")
        
        for class_name in config.CLASS_NAMES:
            icon = config.CLASS_ICONS[class_name]
            color = config.CLASS_COLORS[class_name]
            st.markdown(f"""
                <div style="padding: 0.5rem; margin: 0.5rem 0; background: {color}20; 
                     border-left: 4px solid {color}; border-radius: 5px;">
                    {icon} <strong>{class_name.upper()}</strong>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 💡 How to Use")
        st.markdown("""
        1. Upload an insulator image
        2. Wait for AI analysis
        3. Review the prediction & confidence
        4. Take appropriate action based on results
        """)
        
        st.markdown("---")
        st.markdown("## 🎨 Features")
        st.markdown("""
        - ✅ Real-time classification
        - ✅ High accuracy predictions
        - ✅ Confidence scoring
        - ✅ Professional UI/UX
        - ✅ GPU acceleration support
        """)
        
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); 
                 border-radius: 10px;">
                <small>Developed by Mr. M</small><br>
                <small>© 2026 Insulator Health Monitor</small>
            </div>
        """, unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application function"""
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Load model (auto-train if no trained model found)
    if not st.session_state.model_loaded:
        # First check if a trained model exists (has class_mapping.json)
        temp_classifier = InsulatorClassifier()
        has_trained_model = temp_classifier.is_trained_model()

        if not has_trained_model and not st.session_state.training_done:
            # No trained model — auto-train
            st.warning("⚠️ No trained model found. Starting automatic training...")
            training_success = auto_train_model()
            st.session_state.training_done = True
            if training_success:
                # Clear cached classifier so it reloads the newly trained model
                load_classifier.clear()
                st.rerun()
            else:
                st.error("Training failed. The app will try to use the pretrained model.")

        # Load the model (trained or fallback)
        with st.spinner("🔄 Loading AI model... Please wait..."):
            classifier, success = load_classifier()
            st.session_state.classifier = classifier
            st.session_state.model_loaded = success
            
            if success:
                st.success("✅ Model loaded successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Failed to load model. Please check the model path in config.py")
                st.info(f"Expected model path: `{config.MODEL_PATH}`")
                return
    
    # Main content area
    st.markdown("---")
    
    # Upload section
    st.markdown("## 📤 Upload Insulator Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=config.SUPPORTED_FORMATS,
        help=f"Supported formats: {', '.join(config.SUPPORTED_FORMATS)}"
    )
    
    # Sample images section (optional)
    with st.expander("📁 Or try with sample images from dataset"):
        st.info("You can manually select test images from the `dataset/test/` folder on your local machine.")
    
    # Process uploaded image
    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file)
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📥 Loading image...")
            progress_bar.progress(25)
            time.sleep(0.3)
            
            status_text.text("🔍 Preprocessing...")
            progress_bar.progress(50)
            time.sleep(0.3)
            
            status_text.text("🤖 Running AI model...")
            progress_bar.progress(75)
            
            # Make prediction
            result = st.session_state.classifier.predict(image)
            
            progress_bar.progress(100)
            status_text.text("[OK] Analysis complete!")
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            display_prediction_result(result, image)
            
            # Download results option
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                result_text = f"""
Insulator Health Analysis Report
================================
Predicted Class: {result['class'].upper()}
Confidence: {format_confidence(result['confidence'])}

Detailed Probabilities:
{chr(10).join([f'  - {k.upper()}: {format_confidence(v)}' for k, v in result['all_probabilities'].items()])}

Status: {'⚠️ ALERT - Damage Detected' if result['class'] == 'damage' else '✅ OK - No Damage'}
                """
                
                st.download_button(
                    label="📄 Download Report",
                    data=result_text,
                    file_name=f"insulator_analysis_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
        except Exception as e:
            st.error(f"[ERROR] Error processing image: {str(e)}")
            st.info("Please make sure you uploaded a valid image file.")
    
    else:
        # Show upload prompt
        st.markdown("""
            <div class="upload-section fade-in">
                <h3>👆 Upload an Image to Get Started</h3>
                <p>Drag and drop or click to browse</p>
                <p style="font-size: 0.9rem; color: #666; margin-top: 1rem;">
                    Supported formats: JPG, JPEG, PNG, BMP, TIFF
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Show sample results (optional demo)
        st.markdown("---")
        st.markdown("### 📊 Example Results")
        
        demo_cols = st.columns(2)
        with demo_cols[0]:
            st.markdown("""
                <div style="padding: 1.5rem; background: #FFEBEE; border-radius: 10px; 
                     border-left: 4px solid #FF4B4B;">
                    <h4>⚠️ Damage Detected</h4>
                    <p><strong>Confidence:</strong> 98.5%</p>
                    <p><small>Immediate inspection recommended</small></p>
                </div>
            """, unsafe_allow_html=True)
        
        with demo_cols[1]:
            st.markdown("""
                <div style="padding: 1.5rem; background: #E8F5E9; border-radius: 10px; 
                     border-left: 4px solid #00C853;">
                    <h4>✅ Normal Condition</h4>
                    <p><strong>Confidence:</strong> 99.2%</p>
                    <p><small>No action required</small></p>
                </div>
            """, unsafe_allow_html=True)


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
