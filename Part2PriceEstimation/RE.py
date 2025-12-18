import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import numpy as np

# --- Check for required dependencies ---
try:
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    st.error("❌ Required library 'scikit-learn' is not installed. Please run:\n\n```bash\npip install scikit-learn\n```")
    st.stop()

# Load model and scaler
@st.cache_resource
def load_model():
    return joblib.load("prices.pkl"), joblib.load("scaler.pkl")

model, scaler = load_model()
data = pd.read_csv("RealEstateValidated.csv")

# Questions for chatbot
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

# Prediction function
def run_model(answers):
    try:
        location = answers["What is your location?"].strip().lower()
        house_type = answers["What is your type of house?"].strip()
        sqft = int(answers["How many sqft property is required?"])
        bedrooms = int(answers["How many bedrooms are required?"])
        furnishing = answers["What type of furnishing do you need?"].strip()
        constr_info = answers["What about construction info?"].strip()
        floor = int(answers["Which floor do you need?"])
        facing = answers["What type of facing required for property?"].strip()

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

        # Encode categorical fields
        house_type_map = {'Farm Houses': 3, 'Apartments': 1, 'Houses & Villas': 2, 'Builder Floors': 0}
        row['house_type'] = row['house_type'].map(house_type_map)
        if row['house_type'].isnull().any():
            raise ValueError("Invalid house type. Use: " + ", ".join(house_type_map.keys()))

        data['location'] = data['location'].str.strip().str.lower()
        location_price_dict = data.groupby('location')['price'].mean().round(2).to_dict()
        row['location'] = row['location'].map(location_price_dict)
        if row['location'].isnull().any():
            raise ValueError("Invalid location. Try one listed in your dataset.")

        furnishing_map = {'Semi-Furnished': 1, 'Furnished': 2, 'Unfurnished': 0}
        row['furnishing'] = row['furnishing'].map(furnishing_map)
        if row['furnishing'].isnull().any():
            raise ValueError("Invalid furnishing type.")

        constr_map = {'New Launch': 0, 'Ready to Move': 1, 'Under Construction': 2}
        row['constr_info'] = row['constr_info'].map(constr_map)
        if row['constr_info'].isnull().any():
            raise ValueError("Invalid construction info.")

        facing_map = {
            'North-East': 0, 'South-East': 1, 'East': 2, 'West': 3,
            'South-West': 4, 'North': 5, 'South': 6, 'North-West': 7
        }
        row['facing'] = row['facing'].map(facing_map)
        if row['facing'].isnull().any():
            raise ValueError("Invalid facing direction.")

        # Scale numeric columns
        numeric_columns = ['sqft', 'bedrooms', 'floor_no']
        row[numeric_columns] = scaler.transform(row[numeric_columns])

        # Make prediction
        predicted_price = model.predict(row)[0]
        return int(predicted_price)

    except Exception as e:
        return f"Error during prediction: {e}"

# Display chatbot interface
st.subheader(":orange[AI-Powered Price Prediction Chatbot]")

col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
with col2:
    intro_html = """<div style='text-align: left; margin-bottom: 10px;'>
        <span style='background-color: red; padding: 8px 12px; border-radius: 10px; display: inline-block; color: white;'>
        🤖: Hello! I'm your AI assistant to help you find the best price based on your details. Let's get started!
        </span></div>"""

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
        if isinstance(prediction, (int, float)):
            result_msg = f"🤖: Estimated Price: ₹ {prediction:}"
        else:
            result_msg = f"🤖: {prediction}"
        chat_html += f"""<div style='margin-bottom: 5px;'>
                <span style='background-color: purple; padding: 8px 12px; border-radius: 10px; display: inline-block; color: white;'>{result_msg}</span>
                </div></div>"""

    chat_html += "</div><script>document.getElementById('chat-container').scrollTop = document.getElementById('chat-container').scrollHeight;</script>"
    components.html(intro_html + chat_html, height=400)

# Chat input
cola, colb, colc = st.columns([0.2, 0.6, 0.2])
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
