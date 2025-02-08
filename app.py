from flask import Flask, render_template, request, jsonify
from chatbot import predict_class, get_response
import json
import traceback

app = Flask(__name__)

# Load dataset
with open("data/intents.json") as file:
    data = json.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'message' not in data:
            print("No message received in request")
            return jsonify({"error": "No message provided"}), 400
            
        user_message = data['message']
        print(f"Received message: {user_message}")  # Debug print
        
        intents = predict_class(user_message)
        print(f"Predicted intents: {intents}")  # Debug print
        
        response = get_response(intents)
        print(f"Generated response: {response}")  # Debug print
        
        return jsonify({"response": response})
        
    except Exception as e:
        print(f"Error in chatbot_response: {str(e)}")
        print(traceback.format_exc())  # Print full traceback
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)