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
    base_prompt = f"[INST]Optimize this LLM prompt: '{optimized_prompt}'. Suggest a clearer, more specific version (e.g., add context like audience, length, format, or purpose): [/INST]"
    
    if tokenizer is None or model is None:
        return "Model not loaded. Please restart the app"
    
    logging.debug(f"Base prompt: '{base_prompt}'")

    try:
        inputs = tokenizer(base_prompt, return_tensors="pt").to(model.device)
        logging.debug(f"Tokenized inputs: {inputs}")

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        logging.debug(f"Generated outputs: {outputs}")

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.debug(f"Decoded result: '{result}'")

        # Remove the base prompt from the result (Mistral does add the prompt back in)
        result = result.replace(base_prompt.replace("[INST]", "").replace("[/INST]", ""), "").strip()

        # Remove repetitive phrases
        result = re.sub(r'(\s*\n){2,}', '\n\n', result)
        result = re.sub(r'Write a story\.', '', result)
        result = re.sub(r'Write a story', '', result)

        result = result.strip()

        if result:
            return optimized_prompt + " " + result
        else:
            return optimized_prompt + " (The model was unable to provide a specific optimization, try adding context like audience, length like 'under 50 words', or format)"

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