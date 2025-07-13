import os
import sys
from flask import Flask, request, jsonify
import pandas as pd
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.get_text import GenerateText  
from src.handle_users.handle_users import SignUp, Login
from src.journaling import UpdateJournal
from src.audio_chat import transcribe_audio

# ========== 🔧 Setup ==========
app = Flask(__name__)

# Add CORS headers for frontend requests
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# FIX: Add "text input" column to dummy data
dummy_row = {
    "text input": [""],  # Add this line
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

# Try to initialize generator at startup
try:
    from src.get_text import GenerateText
    generator = GenerateText()
    print("✅ Generator initialized successfully")
except Exception as e:
    print(f"⚠️ Generator initialization failed: {e}")
    generator = None

# ========== 📡 ROUTES ==========

@app.route("/")
def home():
    return jsonify({
        "status": "🧠 Alita API is running",
        "generator_status": "loaded" if generator else "failed"
    })

@app.route("/generate", methods=["POST"])
def generate_response():
    try:
        print("🔄 Processing request to /generate")
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        message = data.get("message", "").strip()
        
        if not message:
            return jsonify({"error": "Missing or empty message"}), 400

        print(f"📩 Received message: {message}")

        if generator is None:
            return jsonify({
                "error": "AI model not available",
                "response": "I'm sorry, my AI model isn't loaded right now. Please try again later."
            }), 503

        params_df = dummy_df.copy()
        params_df.loc[0, "text input"] = message
        
        exclusive_params = params_df.drop(columns=["text input"]).iloc[0].to_dict()
        
        print(f"🔧 Parameters prepared: {list(exclusive_params.keys())}")

        response_text = generator.generate(
            input_text=message,
            exclusive_parameters=exclusive_params
        )

        print(f"🤖 Generated response: {response_text}")

        if response_text is None or str(response_text).strip() == "":
            return jsonify({
                "error": "Empty response generated",
                "response": "I'm having trouble generating a response right now. Could you try rephrasing your question?"
            }), 500

        return jsonify({
            "message": message,
            "response": response_text,
            "status": "success"
        }), 200

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Error in generate_response: {e}")
        print(f"📝 Full traceback: {error_details}")
        
        return jsonify({
            "error": f"Generation failed: {str(e)}",
            "response": "I'm experiencing technical difficulties. Please try again."
        }), 500

# Add debugging endpoint
@app.route("/debug", methods=["GET"])
def debug():
    return jsonify({
        "generator_available": generator is not None,
        "dummy_df_columns": list(dummy_df.columns),
        "dummy_df_shape": dummy_df.shape,
        "sample_data": dummy_df.iloc[0].to_dict()
    })

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
    print("🚀 Starting Alita API...")
    print(f"🌐 Frontend should connect to: http://localhost:8080")
    print(f"🤖 Generator status: {'Loaded' if generator else 'Not available'}")
    app.run(host="0.0.0.0", port=8080, debug=True)
