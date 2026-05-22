import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import shap
from utils.ui import inject_custom_css, render_header
from utils.model import load_model_and_explainer, get_performance_metrics
from utils.ai_explanation import generate_explanation
from utils.data_prep import prepare_applicant_data

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RiskPilot | Loan Analytics",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UI SETUP ---
inject_custom_css()
render_header()

# --- LOAD RESOURCES ---
model, model_features, explainer = load_model_and_explainer()
if model is None:
    st.stop()

# --- APP LAYOUT ---
tab1, tab2 = st.tabs(["Risk Assessment Console", "System Metrics"])

# --- TAB 1: PREDICTION ---
with tab1:
    st.markdown("### New Applicant Risk Profiling")
    st.write("Enter applicant details below to generate a risk assessment.")
    
    with st.form("prediction_form"):
        st.markdown("#### 1. Loan Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            loan_amnt = st.number_input('Requested Amount ($)', min_value=1000, max_value=50000, value=15000, step=500)
            purpose = st.selectbox('Loan Purpose', ['debt_consolidation', 'credit_card', 'home_improvement', 'other'])
        with col2:
            term = st.selectbox('Loan Term (Months)', [36, 60])
            grade = st.selectbox('Internal Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        with col3:
            grade_to_rate = {'A': 7.5, 'B': 10.5, 'C': 13.5, 'D': 16.5, 'E': 19.5, 'F': 22.5, 'G': 25.5}
            int_rate = st.number_input('Interest Rate (%)', min_value=5.0, max_value=35.0, value=grade_to_rate[grade], step=0.1)

        st.markdown("#### 2. Applicant Financials")
        col4, col5, col6 = st.columns(3)
        with col4:
            annual_inc = st.number_input('Annual Income ($)', min_value=0, max_value=2000000, value=75000, step=1000)
            emp_length = st.slider('Employment Length (Years)', 0, 10, 5)
        with col5:
            dti = st.number_input('Debt-to-Income Ratio', min_value=0.0, max_value=100.0, value=20.0, step=1.0)
            home_ownership = st.selectbox('Home Ownership', ['MORTGAGE', 'RENT', 'OWN', 'ANY'])
        with col6:
            fico_score = st.slider('FICO Credit Score', 300, 850, 720)

        st.markdown("#### 3. Credit Profile")
        col7, col8 = st.columns(2)
        with col7:
            revol_bal = st.number_input('Total Revolving Balance ($)', min_value=0, value=15000, step=1000)
            open_acc = st.number_input('Open Credit Lines', min_value=1, max_value=50, value=10, step=1)
        with col8:
            revol_util = st.number_input('Revolving Utilization (%)', min_value=0.0, max_value=150.0, value=50.0, step=1.0)
            total_acc = st.number_input('Total Credit Lines', min_value=2, max_value=100, value=25, step=1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Risk Analysis", use_container_width=True)

    if submitted:
        with st.spinner('Analyzing applicant data via RiskPilot engine...'):
            form_data = {
                'loan_amnt': loan_amnt, 'term': term, 'int_rate': int_rate,
                'grade': grade, 'emp_length': emp_length, 'home_ownership': home_ownership,
                'annual_inc': annual_inc, 'purpose': purpose, 'dti': dti, 
                'fico_score': fico_score, 'revol_util': revol_util, 
                'total_acc': total_acc, 'open_acc': open_acc, 'revol_bal': revol_bal
            }
            
            # Prepare Data using utils
            final_df = prepare_applicant_data(form_data, model_features)
            
            # Prediction
            prediction = model.predict(final_df)[0]
            probability = model.predict_proba(final_df)[0][1]
            
            # SHAP Explanation
            shap_values = explainer.shap_values(final_df)
            feature_names = final_df.columns
            shap_impact = shap_values[0] if isinstance(shap_values, list) else shap_values
            if len(shap_impact.shape) > 1: # Handle multiclass shape
                 shap_impact = shap_impact[0]
            
            # Create dict of top 5 features by absolute SHAP value
            feature_importance = list(zip(feature_names, shap_impact))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            top_features = {k: v for k, v in feature_importance[:5]}

            st.markdown("---")
            st.markdown("### Underwriting Results")
            
            result_tab1, result_tab2, result_tab3 = st.tabs(["Decision Output", "Underwriting Summary", "Risk Driver Analysis"])
            
            with result_tab1:
                if prediction == 1:
                    st.error("**LIKELY TO DEFAULT**\n\nHigh risk detected.")
                else:
                    st.success("**NOT LIKELY TO DEFAULT**\n\nLow risk detected.")
                st.metric("Probability of Default", f"{probability:.1%}")
                
            with result_tab2:
                explanation = generate_explanation(prediction, probability, top_features)
                st.info(explanation)
                
            with result_tab3:
                st.write("This chart illustrates how each financial metric pushed the model's prediction higher (red) or lower (blue) compared to the baseline.")
                
                # Make matplotlib figure professional and smaller
                fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#ffffff')
                
                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                shap_obj = shap.Explanation(values=shap_impact, base_values=base_value, data=final_df.iloc[0], feature_names=feature_names)
                shap.waterfall_plot(shap_obj, show=False)
                
                # Clean up plot spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#cbd5e1')
                ax.spines['left'].set_color('#cbd5e1')
                plt.tight_layout()
                
                st.pyplot(fig, use_container_width=True)

# --- TAB 2: METRICS ---
with tab2:
    st.markdown("### Core Engine Performance")
    st.write("The RiskPilot model (XGBoost Engine) is continuously monitored for accuracy and fairness.")
    
    metrics = get_performance_metrics()
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Overall Accuracy", f"{metrics['accuracy']:.1%}")
    col_m2.metric("Precision (Default Identification)", f"{metrics['precision']:.1%}")
    col_m3.metric("Recall (Default Catch Rate)", f"{metrics['recall']:.1%}")
    col_m4.metric("F1-Score", f"{metrics['f1_score']:.2f}")
    
    st.markdown("---")
    st.write("**Note:** Metrics reflect validation on a held-out test set of 85,000 historical loans. The dataset was preprocessed with SMOTE to handle class imbalance, ensuring high recall for under-represented defaults.")