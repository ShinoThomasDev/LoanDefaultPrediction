import streamlit as st
import joblib
import shap

@st.cache_resource
def load_model_and_explainer():
    """Loads model, features, and creates SHAP explainer."""
    try:
        model = joblib.load('models/xgb_model_engineered.joblib')
        model_features = joblib.load('models/model_features_engineered.joblib')
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
