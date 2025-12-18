
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

# Retrieve Based AI Modules
# pip install sentence-transformers faiss-cpu
import faiss
from sentence_transformers import SentenceTransformer

# Loading Q&A dataset
@st.cache_data
def load_data():
    df = pd.read_csv("final.csv").dropna()
    return df["Question"].astype(str).tolist(), df["Answer"].astype(str).tolist()

# Load SBERT model and encode dataset questions
@st.cache_resource
def load_model_and_index(questions):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(questions)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return model, index, embeddings

# Search logic
def get_best_answer(query, model, index, questions, answers):
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), k=1)
    return answers[I[0][0]]

###################################################### Streamlit app ############################################################

# # Page config (optional dark theme look)
# st.set_page_config(page_title="My Chatbot", layout="centered")

st.subheader(":orange[AI Powered Chatbot:]")
cola, colb , colc = st.columns([0.2,0.05,0.2])
with colb:
    st.write(":red[Converse!]")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "bot", "content": "Hi I am CRM Bot! we are here to help you?"}]
col1, col2, col3 = st.columns([0.2,0.6,0.2])
with col2:
    # Create the complete chat container HTML
    chat_html = """<div id="chat-container" style="height: 300px; overflow-y: auto; border: 1px solid #ccc; border-radius: 10px;
    padding: 10px; background-color: black; margin-bottom: 10px;">"""
    # Add messages inside the container

    for message in st.session_state.messages:
        if message["role"] == "user":
            chat_html += f"""<div style='text-align: right; margin-bottom: 10px;'>
            <span style='background-color: green; padding: 8px 12px; border-radius: 10px; display: inline-block; color: white;'>
            🧑: {message['content'].title()}
            </span>
            </div>"""
        else:
            chat_html += f"""<div style='text-align: left; margin-bottom: 10px;'>
            <span style='background-color: blue; padding: 8px 12px; border-radius: 10px; display: inline-block; color: white;'>
            🤖: {message['content']}
            </span>
            </div>"""

    # Close the chat container
    chat_html += "</div>"

    chat_html += """<script>
    var container = document.getElementById("chat-container");
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
    </script>
    </div>"""

    # Render the full HTML
    components.html(chat_html, height=330, scrolling=False)
    # st.markdown(chat_html, unsafe_allow_html=True)

# Align of chat_input to center using columns
col1, col2, col3 = st.columns([0.2,0.6,0.2])
with col2:
    user_question = st.chat_input("Type your Question:")

# Respond to user input
if user_question:
    # Append user input and bot response to the session state
    st.session_state.messages.append({"role": "user", "content": user_question})
    # Load data and model
    questions, answers = load_data()
    model, index, _ = load_model_and_index(questions)
    answer = get_best_answer(user_question, model, index, questions, answers)
    st.session_state.messages.append({"role": "bot", "content": answer})
    # Clear the input box
    st.rerun()

