import streamlit as st
import anthropic
import os

def generate_explanation(prediction_label, probability, top_features_dict):
    """Uses Claude API to generate a professional banking explanation."""
    api_key = ""
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass
        
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    
    if not api_key:
        # Fallback rule-based generator
        risk_level = "High" if prediction_label == 1 else "Low"
        decision = "Flagged for Default" if prediction_label == 1 else "Likely to Repay"
        
        explanation = f"**Risk Level: {risk_level}**\n\nThe system has classified this application as **{decision}** with a probability of default of {probability:.1%}. \n\n**Primary Driving Factors:**\n"
        
        for feature, impact in top_features_dict.items():
            name = feature.replace('_', ' ').title()
            direction = "increased" if impact > 0 else "decreased"
            explanation += f"- **{name}**: This financial metric significantly {direction} the applicant's assessed risk profile.\n"
            
        explanation += "\n*(Note: Provide a valid Anthropic API Key for a comprehensive underwriting report.)*"
        return explanation

    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        prompt = f"""
        You are an elite automated risk underwriting system at a top-tier commercial bank. Generate a structured, highly professional risk assessment summary for a human underwriter.

        [SYSTEM INPUTS]
        - System Decision: {'DECLINE (High Default Risk)' if prediction_label == 1 else 'APPROVE (Low Default Risk)'}
        - Computed Probability of Default: {probability:.1%}
        - Primary Driving Metrics (Positive values increase default risk, negative values decrease risk):
        {top_features_dict}

        [OUTPUT REQUIREMENTS]
        Write a concise, structured assessment using the exact format below (do not include markdown headers, just bold text):
        
        **Risk Assessment:** (1 sentence summary of the decision and risk level)
        
        **Key Drivers:** 
        - (Bullet point 1 explaining the strongest metric)
        - (Bullet point 2 explaining the next strongest metric)
        
        **Underwriting Recommendation:** (1 formal sentence advising the bank on how to proceed)

        Maintain an objective, formal banking tone. Do not mention "SHAP", "machine learning", or "AI" in your output. Treat the inputs as factual financial metrics.
        """
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            temperature=0.3,
            system="You are a professional banking risk system.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error communicating with Claude API: {e}. Please check your API key."
