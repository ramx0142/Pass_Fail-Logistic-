import streamlit as st
import joblib
import numpy as np

# 1. Load the trained model
# Make sure 'student_model.pkl' is in the same folder as this app.py file
try:
    model = joblib.load('student_exam_data.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please export 'student_model.pkl' from your notebook and place it here.")
    st.stop()

# 2. App Title and Description
st.set_page_config(page_title="Student Exam Predictor", page_icon="🎓")

st.title("🎓 Student Pass/Fail Predictor")
st.write("Enter the study hours and previous exam score to predict if the student will Pass or Fail.")

# 3. Input Fields
col1, col2 = st.columns(2)

with col1:
    # Based on your data, study hours can be decimals (e.g. 6.78)
    study_hours = st.number_input(
        "Study Hours", 
        min_value=0.0, 
        max_value=24.0, 
        value=5.0, 
        step=0.5,
        format="%.2f"
    )

with col2:
    # Previous exam scores range from ~40 to 100 in your data
    prev_score = st.number_input(
        "Previous Exam Score", 
        min_value=0.0, 
        max_value=100.0, 
        value=70.0, 
        step=1.0,
        format="%.2f"
    )

# 4. Prediction Logic
if st.button("Predict Result"):
    # The model expects input in the order: ['Study Hours', 'Previous Exam Score']
    input_data = np.array([[study_hours, prev_score]])
    
    try:
        prediction = model.predict(input_data)
        result = prediction[0]
        
        st.markdown("---")
        st.subheader("Prediction:")
        
        # 1 means Pass, 0 means Fail
        if result == 1:
            st.success("🎉 The Student will likely **PASS**.")
        else:
            st.error("⚠️ The Student will likely **FAIL**.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Footer
st.caption("Model: Logistic Regression")
