"""
Flask web application for TinyLLM.
Provides a simple web interface for interacting with the model.
"""

from flask import Flask, render_template, request, jsonify
import torch
import config
from vocab import Vocabulary
from model import TinyLLMModel
from generate import generate_response
import os

app = Flask(__name__)

# Global variables for model and vocab
model = None
vocab = None


def load_model():
    """Load the trained model and vocabulary."""
    global model, vocab
    
    # Load vocabulary
    if os.path.exists(config.VOCAB_SAVE_PATH):
        vocab = Vocabulary.load(config.VOCAB_SAVE_PATH)
    else:
        vocab = Vocabulary()
        vocab.save(config.VOCAB_SAVE_PATH)
    
    config.VOCAB_SIZE = vocab.size
    
    # Initialize model
    model = TinyLLMModel(vocab_size=vocab.size)
    
    # Load trained weights if available
    if os.path.exists(config.MODEL_SAVE_PATH):
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded from {config.MODEL_SAVE_PATH}")
    else:
        print("Warning: No trained model found. Please train the model first using run.py")
        model.eval()
    
    # Move to CPU
    device = torch.device(config.DEVICE)
    model = model.to(device)


@app.route('/')
def index():
    """Render the main page."""
    return render_template('retro.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Generate a response to a user prompt."""
    try:
        data = request.json
        prompt = data.get('prompt', '').strip()
        
        if not prompt:
            return jsonify({'error': 'Please provide a prompt'}), 400
        
        if model is None or vocab is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Generate response
        response = generate_response(prompt, model, vocab)
        
        return jsonify({
            'prompt': prompt,
            'response': response
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Loading TinyLLM model...")
    load_model()
    print("Model loaded! Starting web server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)