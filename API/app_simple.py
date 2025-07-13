import os
import sys
from flask import Flask, request, jsonify
import pandas as pd
import traceback

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

print(f"📁 Current directory: {current_dir}")
print(f"📁 Parent directory: {parent_dir}")

# Try imports with error handling
try:
    from src.get_text import GenerateText
    generator = GenerateText()
    print("✅ Generator loaded successfully")
except Exception as e:
    print(f"⚠️ Generator failed to load: {e}")
    generator = None

# Simple mock generator for testing
class MockGenerator:
    def generate(self, input_text, exclusive_parameters=None):
        return f"Mock response to: {input_text}"

# ========== 🔧 Setup ==========
app = Flask(__name__)

# Add CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Handle preflight requests
@app.route("/generate", methods=["OPTIONS"])
def generate_options():
    return '', 200

# Dummy patient data
dummy_row = {
    "text input": [""],
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
    return jsonify({
        "status": "🧠 Alita API is running",
        "generator_status": "loaded" if generator else "mock",
        "port": 8080,
        "endpoints": ["/generate", "/health", "/debug"]
    })

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "generator": generator is not None})

@app.route("/generate", methods=["POST"])
def generate_response():
    try:
        print("📩 Received generate request")
        
        data = request.get_json()
        if not data:
            print("❌ No JSON data provided")
            return jsonify({"error": "No JSON data provided"}), 400
            
        message = data.get("message", "").strip()
        if not message:
            print("❌ Empty message")
            return jsonify({"error": "Missing or empty message"}), 400

        print(f"📝 Processing message: {message}")

        # Use generator or fallback to mock
        current_generator = generator if generator else MockGenerator()
        
        # Prepare parameters
        params_df = dummy_df.copy()
        params_df.loc[0, "text input"] = message
        exclusive_params = params_df.drop(columns=["text input"]).iloc[0].to_dict()
        
        print(f"🔧 Using parameters: {list(exclusive_params.keys())}")

        # Generate response
        response_text = current_generator.generate(
            input_text=message,
            exclusive_parameters=exclusive_params
        )

        print(f"🤖 Generated: {response_text}")

        if not response_text:
            return jsonify({
                "error": "Empty response generated",
                "response": "I'm having trouble generating a response. Please try again."
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

@app.route("/debug")
def debug():
    return jsonify({
        "generator_available": generator is not None,
        "dummy_df_columns": list(dummy_df.columns),
        "dummy_df_shape": dummy_df.shape,
        "sample_data": dummy_df.iloc[0].to_dict(),
        "python_path": sys.path[:3]
    })

# ========== 🏁 Start Server ==========
if __name__ == "__main__":
    print("🚀 Starting Alita API...")
    print(f"🌐 Frontend should connect to: http://localhost:5001")
    print(f"🤖 Generator status: {'Loaded' if generator else 'Mock mode'}")
    print("📡 Available endpoints:")
    print("   - GET  /")
    print("   - GET  /health") 
    print("   - POST /generate")
    print("   - GET  /debug")
    
    app.run(host="0.0.0.0", port=5001, debug=True)
