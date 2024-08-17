import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model and tokenizer
model_name = "bert-base-uncased"
model_path = "disaster_model.pth"
tokenizer_path = "tokenizer/"

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Define the disaster mapping
disaster_mapping = {
    0: 'Drought',
    1: 'Wildfire',
    2: 'Earthquake',
    3: 'Floods',
    4: 'Hurricanes',
    5: 'Tornadoes'
}

# Custom CSS to style the text area and other elements
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 700px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #2b2b2b;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
    }
    .stTextArea>div>div>textarea {
        background-color: #1e1e1e;
        border: 1px solid #555;
        border-radius: 5px;
        font-size: 1.1rem;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff6347;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 5px;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #ff4500;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    p {
        color: #dcdcdc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Disaster Prediction Application")
st.write("Enter a text about a disaster to predict the type of disaster.")

# Text input
user_input = st.text_area("Enter text here:", "", height=200)

# Prediction button
if st.button("Predict"):
    if user_input:
        # Tokenize the input text
        new_encodings = tokenizer([user_input], truncation=True, padding=True, return_tensors="pt")
        
        # Perform prediction
        with torch.no_grad():
            outputs = model(**new_encodings)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Get the predicted disaster name
        predicted_disaster = disaster_mapping[predictions.item()]
        
        # Display the result
        st.write(f"Predicted Disaster: **{predicted_disaster}**")
    else:
        st.write("Please enter some text for prediction.")
