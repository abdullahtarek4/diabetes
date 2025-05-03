import streamlit as st 
import numpy as np
import joblib

# Load your trained model and scaler
model = joblib.load(open('opt.pkl', 'rb'))
scaler = joblib.load(open('scaler_10features.pkl', 'rb'))  # Load the scaler used during training

# All 14 features you want to collect from the user
all_inputs = [
    'Age', 'BMI', 'Waist_Circumference', 'Fasting_Blood_Glucose', 'HbA1c',
    'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 'GGT', 'Serum_Urate',
    'Physical_Activity_Level', 'Dietary_Intake_Calories',
    'Alcohol_Consumption', 'Smoking_Status', 'Family_History_of_Diabetes'
]

# ✅ Only the features the model was trained on
model_features = [
    'Age', 'BMI', 'Waist_Circumference', 'Fasting_Blood_Glucose', 'HbA1c',
    'Blood_Pressure_Systolic', 'GGT', 'Serum_Urate',
    'Alcohol_Consumption', 'Family_History_of_Diabetes'
]

def main():
    st.set_page_config(layout='wide')
    st.title('Diabetes Prediction App (Trained on 10 Features)')

    inputs = {}
    for feature in all_inputs:
        if feature in ['Physical_Activity_Level', 'Alcohol_Consumption', 'Smoking_Status', 'Family_History_of_Diabetes']:
            st.markdown(f"**{feature.replace('_', ' ')}**")
            if feature == 'Physical_Activity_Level':
                inputs[feature] = st.selectbox(feature, [0, 1, 2], format_func=lambda x: ["Low", "Moderate", "High"][x])
            elif feature == 'Smoking_Status':
                inputs[feature] = st.selectbox(feature, [0, 1, 2], format_func=lambda x: ["Never", "Former", "Current"][x])
            elif feature == 'Alcohol_Consumption':
                inputs[feature] = st.selectbox(feature, [0, 1, 2], format_func=lambda x: ["No", "Moderate", "Heavy"][x])
            elif feature == 'Family_History_of_Diabetes':
                inputs[feature] = st.radio(feature, [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        else:
            inputs[feature] = st.number_input(f'Enter {feature}:', step=0.1)

    if st.button('Predict'):
        try:
            # ✅ Extract only the features the model expects
            input_values = np.array([inputs[feature] for feature in model_features]).reshape(1, -1)

            # ✅ Apply scaling
            scaled_input = scaler.transform(input_values)

            
            

            # ✅ Predict with the model
            prediction = model.predict(scaled_input)

            if prediction[0] == 1:
                st.error("⚠️ The model predicts a high risk of Diabetes. Please consult a doctor.")
            else:
                st.success("✅ The model predicts a low risk of Diabetes. Keep maintaining a healthy lifestyle.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
