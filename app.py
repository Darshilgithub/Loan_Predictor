# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and scaler
@st.cache_resource
def load_model():
    with open('loan_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("💰 Loan Approval Prediction System")
st.markdown("### Check your loan eligibility instantly!")
st.markdown("---")

try:
    model, scaler, feature_names = load_model()
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"])
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    with col2:
        st.subheader("Financial Information")
        applicant_income = st.number_input(
            "Applicant Income (₹)", 
            min_value=0, 
            max_value=100000, 
            value=5000,
            step=500
        )
        coapplicant_income = st.number_input(
            "Co-applicant Income (₹)", 
            min_value=0, 
            max_value=100000, 
            value=0,
            step=500
        )
        loan_amount = st.number_input(
            "Loan Amount (₹ in thousands)", 
            min_value=0, 
            max_value=1000, 
            value=100,
            step=10
        )
        loan_amount_term = st.selectbox(
            "Loan Term (months)", 
            [360, 180, 120, 240, 300]
        )
        credit_history = st.selectbox(
            "Credit History (1 = Good, 0 = Bad)", 
            [1.0, 0.0]
        )
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔍 Check Loan Eligibility", use_container_width=True):
        # Encode inputs
        gender_encoded = 1 if gender == "Male" else 0
        married_encoded = 1 if married == "Yes" else 0
        
        dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
        dependents_encoded = dependents_map[dependents]
        
        education_encoded = 0 if education == "Graduate" else 1
        self_employed_encoded = 1 if self_employed == "Yes" else 0
        
        property_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
        property_encoded = property_map[property_area]
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'Gender': [gender_encoded],
            'Married': [married_encoded],
            'Dependents': [dependents_encoded],
            'Education': [education_encoded],
            'Self_Employed': [self_employed_encoded],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_encoded]
        })
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        st.markdown("---")
        
        # Display result
        if prediction[0] == 1:
            st.success("✅ Congratulations! Your loan is likely to be APPROVED")
            st.balloons()
            confidence = prediction_proba[0][1] * 100
            st.metric("Approval Confidence", f"{confidence:.2f}%")
        else:
            st.error("❌ Sorry! Your loan is likely to be REJECTED")
            confidence = prediction_proba[0][0] * 100
            st.metric("Rejection Confidence", f"{confidence:.2f}%")
        
        # Display input summary
        st.markdown("---")
        st.subheader("Application Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.write(f"**Gender:** {gender}")
            st.write(f"**Married:** {married}")
            st.write(f"**Dependents:** {dependents}")
            st.write(f"**Education:** {education}")
            st.write(f"**Self Employed:** {self_employed}")
            st.write(f"**Property Area:** {property_area}")
        
        with summary_col2:
            st.write(f"**Applicant Income:** ₹{applicant_income:,}")
            st.write(f"**Co-applicant Income:** ₹{coapplicant_income:,}")
            st.write(f"**Total Income:** ₹{applicant_income + coapplicant_income:,}")
            st.write(f"**Loan Amount:** ₹{loan_amount * 1000:,}")
            st.write(f"**Loan Term:** {loan_amount_term} months")
            st.write(f"**Credit History:** {'Good' if credit_history == 1.0 else 'Bad'}")

except FileNotFoundError:
    st.error("⚠️ Model files not found! Please train the model first by running 'train_model.py'")
    st.info("Run the training script to generate the required model files.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Loan Approval Prediction System | ML College Project</p>",
    unsafe_allow_html=True
)