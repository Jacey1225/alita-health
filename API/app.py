import os
import torch
from flask import Flask, request, jsonify
from transformers import T5Tokenizer
from src.data_processing import DataProcessor
from src.model_setup import ModelSetup
import pandas as pd
from flask_cors import CORS
import logging

# ========== 🔧 Setup ==========
app = Flask(__name__)
CORS(app)  # allow frontend/mobile access
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Dummy row for neutral well-being context (not medical)
dummy_row = {
    "text input": ["Hey, I’m just feeling off today."],
    "desired response": ["That’s okay. I'm here to listen."],
    "Age": [18],
    "Gender": ["female"],
    "Mood Score (1-10)": [4],
    "Sleep Quality (1-10)": [3],
    "Physical Activity (hrs/week)": [2],
    "Stress Level (1-10)": [7],
    "Energy Level (1-10)": [5],
    "Social Interaction (1-10)": [6],
    "Recent Journal Entry Length": [120]
}
dummy_df = pd.DataFrame(dummy_row)

# Initialize processor and model
processor = DataProcessor(dummy_df)
model = ModelSetup(processor)
try:
    model.load_state_dict(torch.load("model/medical_t5_model.pth", map_location=device))
except Exception as e:
    logging.error("Failed to load model weights.")
    raise e

model.to(device)
model.eval()

# ========== 📡 ROUTES ==========

@app.route("/")
def home():
    return "🧠 Alita Emotional Support API is running."

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/generate", methods=["POST"])
def generate_response():
    data = request.get_json()

    message = data.get("message", "")
    if not message:
        return jsonify({"error": "Missing message"}), 400

    try:
        logging.info(f"User input: {message}")

        # Build input dataframe
        input_data = dummy_df.copy()
        input_data.loc[0, "text input"] = message

        # Optionally override neutral fields
        for col in input_data.columns:
            if col in data and col not in ["text input", "desired response"]:
                input_data.loc[0, col] = data[col]

        # Preprocess
        user_processor = DataProcessor(input_data)
        input_ids, attention_mask = user_processor.process_text_input()
        numerical_features = user_processor.process_numerical_features()
        numerical_features = user_processor.scale_numerical_features()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        numerical_features = numerical_features.to(device)

        # Model forward
        with torch.no_grad():
            output_dict = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_features=numerical_features,
                labels=None
            )

            final_state = output_dict["final state"]
            attention_mask = output_dict["attention mask"]

            generated_ids = model.t5_model.generate(
                encoder_outputs=(final_state,),
                attention_mask=attention_mask,
                max_length=150,
                num_beams=5,
                early_stopping=True
            )

            response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        logging.info(f"Generated response: {response_text}")
        return jsonify({
            "message": message,
            "response": response_text
        })

    except Exception as e:
        logging.error(f"Error in generate_response: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ========== 🏁 Run Server ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
