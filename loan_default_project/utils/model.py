import os
import streamlit as st
import joblib
import shap

@st.cache_resource
def load_model_and_explainer():
    """Loads model, features, and creates SHAP explainer."""
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, 'models', 'xgb_model_engineered.joblib')
        features_path = os.path.join(base_dir, 'models', 'model_features_engineered.joblib')
        
        model = joblib.load(model_path)
        model_features = joblib.load(features_path)
        explainer = shap.TreeExplainer(model)
        return model, model_features, explainer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def get_performance_metrics():
    """Returns hardcoded performance metrics."""
    return {
        'accuracy': 0.86,
        'precision': 0.79,
        'recall': 0.82,
        'f1_score': 0.80
    }
