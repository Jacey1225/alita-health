import os
import torch
from flask import Flask, request, jsonify
from transformers import T5Tokenizer
from src.data_processing import DataProcessor
from src.model_setup import ModelSetup
import pandas as pd

# ========== 🔧 Setup ==========
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer & model
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Dummy row to help initialize processor and model
dummy_row = {
    "text input": ["Hello, I'm not feeling well."],
    "desired response": ["That's okay, let's talk about it."],
    "Age": [18],
    "Gender": ["female"],
    "Diagnosis": ["general anxiety"],
    "Symptom Severity (1-10)": [7],
    "Mood Score (1-10)": [4],
    "Sleep Quality (1-10)": [3],
    "Physical Activity (hrs/week)": [2],
    "Medication": ["none"],
    "Therapy Type": ["CBT"],
    "Treatment Start Date": ["2024-01-01"],
    "Treatment Duration (weeks)": [5],
    "Stress Level (1-10)": [8],
    "Outcome": ["in progress"],
    "Treatment Progress (1-10)": [5],
    "AI-Detected Emotional State": ["stressed"],
    "Adherence to Treatment (%)": [80]
}
dummy_df = pd.DataFrame(dummy_row)

# Initialize processor and model
processor = DataProcessor(dummy_df)
model = ModelSetup(processor)
model.load_state_dict(torch.load("model/medical_t5_model.pth", map_location=device))
model.to(device)
model.eval()

# ========== 📡 ROUTES ==========

@app.route("/")
def home():
    return "🧠 Alita API is running."

@app.route("/generate", methods=["POST"])
def generate_response():
    data = request.get_json()

    # === 1. Input validation ===
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "Missing message"}), 400

    # === 2. Prepare input ===
    try:
        # Build input dataframe (1-row)
        input_data = dummy_df.copy()
        input_data.loc[0, "text input"] = message

        # Optionally override any numerical fields if sent in JSON
        for col in input_data.columns:
            if col in data and col != "text input" and col != "desired response":
                input_data.loc[0, col] = data[col]

        # Process input
        user_processor = DataProcessor(input_data)
        input_ids, attention_mask = user_processor.process_text_input()
        numerical_features = user_processor.process_numerical_features()
        numerical_features = user_processor.scale_numerical_features()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        numerical_features = numerical_features.to(device)

        # === 3. Forward pass ===
        with torch.no_grad():
            output_dict = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_features=numerical_features,
                labels=None  # Inference mode
            )

            final_state = output_dict["final state"]
            attention_mask = output_dict["attention mask"]

            # Generate response using decoder
            generated_ids = model.t5_model.generate(
                encoder_outputs=(final_state,),
                attention_mask=attention_mask,
                max_length=150,
                num_beams=5,
                early_stopping=True
            )
            response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return jsonify({
            "message": message,
            "response": response_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== 🏁 Run Server ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
