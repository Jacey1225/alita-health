import os
from flask import Flask, request, jsonify
import pandas as pd
from src.get_text import GenerateText  
from src.handle_users.handle_users import SignUp, Login
from src.journaling import UpdateJournal

# ========== 🔧 Setup ==========
app = Flask(__name__)

# Dummy row 
dummy_row = {
    "text input": ["Hello, I'm not feeling well."],
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

# ========== 📡 ROUTES ==========

@app.route("/")
def home():
    return "🧠 Alita API is running."

@app.route("/generate", methods=["POST"])
def generate_response():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Missing message"}), 400

    try:
        input_data = dummy_df.copy()
        input_data.loc[0, "text input"] = message

        # Your friend's generator handles generation & decoding
        generator = GenerateText()
        response_text = generator.generate(
            input_text=message,
            exclusive_parameters=input_data.drop(columns=["text input"]).iloc[0].to_dict()
        )

        return jsonify({
            "message": message,
            "response": response_text
        })

    except Exception as e:
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    survey_data = data.get("survey_data")

    try:
        handler = SignUp(username, password, survey_data)
        handler.add_survey_data()
        handler.save()
        return jsonify({"message": f"User '{username}' signed up successfully."}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    handler = Login(username, password)
    if handler.authenticate():
        user_data = handler.get_user_data()
        if user_data:
            return jsonify({
                "message": f"User '{username}' logged in successfully.",
                "data": user_data
            }), 200
        else:
            return jsonify({"error": "User data not found"}), 404
    else:
        return jsonify({"error": "Invalid credentials"}), 401

@app.route("/update_journal", methods=["POST"])
def update_journal():
    data = request.get_json()
    username = data.get("username")
    entry = data.get("entry")

    try:
        handler = UpdateJournal(username, entry)
        handler.add_entry()
        handler.save()
        return jsonify({"message": f"Journal updated for user '{username}'."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== 🏁 Run Server ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
