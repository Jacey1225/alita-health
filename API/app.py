import os
from flask import Flask, request, jsonify
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.get_text import GenerateText  
from src.handle_users.handle_users import SignUp, Login
from src.journaling import UpdateJournal
from src.audio_chat import transcribe_audio

# ========== 🔧 Setup ==========
app = Flask(__name__)

# Dummy row 
dummy_row = {
    "Age": [28],
    "Gender": ["Male"],
    "Diagnosis": ["Generalized Anxiety Disorder"],
    "Symptom Severity (1-10)": [6],
    "Mood Score (1-10)": [5],
    "Sleep Quality (1-10)": [4],
    "Physical Activity (hrs/week)": [3],
    "Medication": ["Alprazolam"],
    "Therapy Type": ["Exposure Therapy"],
    "Treatment Start Date": ["2024-02-01"],
    "Treatment Duration (weeks)": [8],
    "Stress Level (1-10)": [7],
    "Outcome": ["Stable"],
    "Treatment Progress (1-10)": [6],
    "AI-Detected Emotional State": ["Anxious"],
    "Adherence to Treatment (%)": [90]
}
dummy_df = pd.DataFrame(dummy_row)

# ========== 📡 ROUTES ==========

@app.route("/")
def home():
    return "🧠 Alita API is running."

@app.route("/generate", methods=["POST"])
def generate_response():
    data = request.get_json()
    message = data.get("message").strip()

    if not message:
        return jsonify({"error": "Missing message"}), 400

    try:
        exc_params = dummy_df.copy()

        # Your friend's generator handles generation & decoding
        generator = GenerateText()
        response_text = generator.generate(
            input_text=message,
            exclusive_parameters=exc_params.iloc[0].to_dict()
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
        if survey_data is not None:
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

@app.route("/update-journal", methods=["POST"])
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
    
@app.route("/transcribe-audio", methods=["POST"])
def transcribe_audio_route():
    data = request.get_json()
    listen = data.get("listen", False)

    transcription = transcribe_audio(listen=listen)
    generator = GenerateText()
    response_text = generator.generate(
        input_text=transcription,
        exclusive_parameters=dummy_df.drop(columns=["text input"]).iloc[0].to_dict()
    )
    if transcription:
        return jsonify({
            "transcription": transcription,
            "response": response_text
        }), 200
    else:
        return jsonify({"error": "Transcription failed"}), 500

# ========== 🏁 Run Server ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
