import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="lcsekar/tourism-project-model", filename="best_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Visit with Us - Purchase Prediction App")
st.write("""
This application predicts whether the user will buy a tourism package from Visit with Us
based on user details such as age, occupation, gender, marital status and income.
Please enter the app details below to get the prediction.
""")

# User input
customer_id = st.number_input("Customer ID", min_value=1)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=20, value=2)
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Divorced", "Married", "Unmarried"])
num_trips = st.number_input("Number of Trips", min_value=0, max_value=50, value=5)
passport = st.selectbox("Passport", ["Yes", "No"])
own_car = st.selectbox("Own Car", ["Yes", "No"])
num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income (USD)", min_value=1000, max_value=100000, value=5000)

pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
product_pitched = st.selectbox("Product Pitched", ["Super Deluxe", "Deluxe", "Standard", "Basic", "King"])
num_followups = st.number_input("Number of Followups", min_value=0, max_value=20, value=2)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=30)

# # Assemble input into DataFrame
# input_data = pd.DataFrame([{
#     'app_category': app_category,
#     'free_or_paid': free_or_paid,
#     'content_rating': content_rating,
#     'screentime_category': screentime_category,
#     'app_size_in_mb': app_size,
#     'price_in_usd': price,
#     'number_of_installs': installs,
#     'average_screen_time': screen_time,
#     'active_users': active_users,
#     'no_of_short_ads_per_hour': short_ads,
#     'no_of_long_ads_per_hour': long_ads
# }])

# # Predict button
# if st.button("Predict Revenue"):
#     prediction = model.predict(input_data)[0]
#     st.subheader("Prediction Result:")
#     st.success(f"Estimated Ad Revenue: **${prediction:,.2f} USD**")
