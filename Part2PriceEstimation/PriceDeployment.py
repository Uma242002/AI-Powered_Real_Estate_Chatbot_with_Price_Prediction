
import streamlit as st
import streamlit.components.v1 as components
import joblib

import pandas as pd
import numpy as np

# Load predictive model & Its Supoorting files
@st.cache_resource
def load_model():
    return joblib.load("prices.pkl"), joblib.load("scaler.pkl")

model, scaler = load_model()
data = pd.read_csv("RealEstateValidated.csv") 
# Questions to ask
questions = [
    "What is your location?",
    "What is your type of house?",
    "How many sqft property is required?",
    "How many bedrooms are required?",
    "What type of furnishing do you need?",
    "What about construction info?",
    "Which floor do you need?",
    "What type of facing required for property?"

]
# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "completed" not in st.session_state:
    st.session_state.completed = False

# Function to run prediction
def run_model(answers):
    try:
        location = answers["What is your location?"]
        house_type = answers["What is your type of house?"]
        sqft = int(answers["How many sqft property is required?"])
        bedrooms = int(answers["How many bedrooms are required?"])
        furnishing = answers["What type of furnishing do you need?"]
        constr_info = answers["What about construction info?"]
        floor = int(answers["Which floor do you need?"])
        facing = answers["What type of facing required for property?"]

        details = {
                      'location': location,
                      'house_type': house_type,
                      'sqft': sqft,
                      'bedrooms': bedrooms,
                      'furnishing': furnishing,
                      'constr_info': constr_info,
                      'floor_no': floor,
                      'facing': facing
}
        row = pd.DataFrame([details])
        # Encode categorical values
        row['house_type'].replace({'Farm Houses':3, 'Apartments':1, 'Houses & Villas':2, 'Builder Floors':0}, inplace=True)
        row['location'] = row['location'].str.strip().str.lower()

        # Step 2: Compute mean price for each location
        location_mean_prices = data.groupby('location')['price'].mean().round(2)

        # Step 3: Convert to dictionary
        location_price_dict = location_mean_prices.to_dict()

        # Step 4: Replace location names in X with corresponding mean prices
        row['location'] = row['location'].map(location_price_dict)
        
        row['furnishing'].replace({'Semi-Furnished':1, 'Furnished':2, 'Unfurnished':0}, inplace=True)
        row['constr_info'].replace({'New Launch': 0, 'Ready to Move': 1, 'Under Construction': 2}, inplace=True)
        row['facing'].replace({'North-East': 0, 'South-East': 1, 'East': 2, 'West': 3, 'South-West': 4, 'North': 5, 'South': 6, 'North-West': 7}, inplace=True)

        numeric_columns = ['sqft', 'bedrooms', 'floor_no']
        row[numeric_columns] = scaler.transform(row[numeric_columns])
        
        proba = model.predict(row)[0]
        return proba
    except Exception as e:
        return f"Error during prediction: {e}"
# Header
st.subheader(":orange[AI-Powered Price Prediction Chatbot]")
col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
with col2:

    # Intro message outside scrollable container to keep it pinned
    intro_html = """<div style='text-align: left; margin-bottom: 10px;'>
        <span style='background-color: red; padding: 8px 12px; border-radius: 10px; display: inline-block; color: white;'>
        🤖: Hello! I'm your AI assistant to help you find the best price based on your details. Let's get started!
        </span>
        </div>"""

    chat_html = """<div id="chat-container" style="height: 300px; overflow-y: auto; border: 1px solid #ccc;
    border-radius: 10px; padding: 10px; background-color: black; margin-bottom: 10px;">"""

    for i in range(st.session_state.step):
        q = questions[i]
        a = st.session_state.answers.get(q, "")
        chat_html += f"""<div style='text-align: left; margin-bottom: 10px;'>
            <span style='background-color: blue; padding: 8px 12px; border-radius: 10px; display: inline-block; color: white;'>🤖: {q}</span>
            </div>
            <div style='text-align: right; margin-bottom: 10px;'>
            <span style='background-color: green; padding: 8px 12px; border-radius: 10px; display: inline-block; color: white;'>🧑: {a}</span>
            </div>"""

    if not st.session_state.completed:
        current_question = questions[st.session_state.step]
        chat_html += f"""<div style='text-align: left; margin-bottom: 10px;'>
            <span style='background-color: blue; padding: 8px 12px; border-radius: 10px; display: inline-block; color: white;'>🤖: {current_question}</span>
            </div>"""
    else:
        prediction = run_model(st.session_state.answers)
        chat_html += "<div style='text-align: left; margin-bottom: 10px;'>"
        chat_html += f"""<div style='margin-bottom: 5px;'>
            <span style='background-color: purple; padding: 8px 12px; border-radius: 10px; display: inline-block; color: white;'>🤖: Estimated Price:  {int(prediction):}</span>
            </div>"""
        chat_html += "</div>"

    chat_html += """</div><script>document.getElementById('chat-container').scrollTop = document.getElementById('chat-container').scrollHeight;</script>"""
    components.html(intro_html + chat_html, height=400)
# Chat input
cola, colb, colc = st.columns([0.2,0.6,0.2])
with colb:
    if not st.session_state.completed:
        user_input = st.chat_input("Your answer:")
        if user_input:
            q = questions[st.session_state.step]
            st.session_state.answers[q] = user_input
            st.session_state.step += 1
            if st.session_state.step == len(questions):
                st.session_state.completed = True
            st.rerun()
