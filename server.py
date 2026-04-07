from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from app import agent

app = Flask(__name__)
# Max file size 50MB
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_prompt = data.get('message', '')
    
    if not user_prompt:
        return jsonify({"error": "Empty message."}), 400
        
    try:
        # We simply pass the prompt to the CodeAgent
        # The agent's reasoning loop is internally synchronous and will yield the final answer.
        # Ensure we return it as a string
        final_answer = agent.run(user_prompt)
        return jsonify({"response": str(final_answer)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        # We return the absolute filepath so the frontend can append it to the chat prompt automatically.
        return jsonify({'filepath': filepath, 'filename': filename})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
