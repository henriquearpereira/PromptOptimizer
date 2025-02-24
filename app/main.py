from flask import Flask, request, jsonify, render_template
import torch
import logging
import re
import os
from models.model_loader import get_tokenizer, get_model

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__, static_folder='../static')

tokenizer = get_tokenizer()
model = get_model()

if tokenizer is None or model is None:
    logging.error("Tokenizer or model failed to load. The application may not function correctly.")


def optimize_prompt(prompt):
    logging.debug(f"Starting optimize_prompt with prompt: '{prompt}'")

    if not prompt or prompt.strip() == "":
        logging.debug("Prompt is empty or whitespace.")
        return "Please enter a prompt to optimize."

    optimized_prompt = prompt.strip()
    # Create a two-part prompt structure
    base_prompt = (
        "You are a prompt optimization expert. For the following writing prompt:\n"
        f"'{optimized_prompt}'\n\n"
        "Provide two parts:\n"
        "1. First, give an optimized version of the core prompt that maintains its essence while making it more specific and engaging.\n"
        "2. Then, provide supporting details like:\n"
        "   - Target audience\n"
        "   - Length requirements\n"
        "   - Style/tone\n"
        "   - Key elements to include\n\n"
        "Format your response as:\n"
        "Optimized prompt: [your improved prompt]\n"
        "Supporting details:\n"
        "[list the details]\n\n"
        "Response:"
    )
    
    if tokenizer is None or model is None:
        return "Model not loaded. Please restart the app"
    
    logging.debug(f"Base prompt: '{base_prompt}'")

    try:
        inputs = tokenizer(base_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,  # Increased for two-part response
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response but preserve the two-part structure
        result = result.replace(base_prompt, "").strip()
        result = re.sub(r'Response:', '', result)
        result = re.sub(r'## INPUT ##', '', result)
        result = re.sub(r'## OUTPUT ##', '', result)
        
        # Clean up any extra whitespace while preserving structure
        result = re.sub(r'\s{3,}', '\n\n', result)
        result = result.strip()

        if result:
            return result
        else:
            return prompt + " (Please add more context and specificity)"

    except Exception as e:
        logging.error(f"Error generating prompt: {e}")
        return "An error occurred while optimizing the prompt. Please try again."

@app.route('/', methods=['GET', 'POST'])
def home():
    logging.debug("Home route accessed")
    if request.method == 'POST':
        user_prompt = request.form.get('prompt')
        logging.debug(f"User prompt: {user_prompt}")
        optimized = optimize_prompt(user_prompt)
        logging.debug(f"Optimized prompt: {optimized}")
        return render_template('index.html', prompt=user_prompt, optimized=optimized)
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided"}), 400
    optimized = optimize_prompt(data['prompt'])
    return jsonify({"original": data['prompt'], "optimized": optimized})

if __name__ == '__main__':
    app.run(debug=False, port=5001)