# app/main.py
from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Load a pre-trained GPT-2 model for text generation
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a pipeline for text generation
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=50, temperature=0.7)

# Function to optimize prompts using GPT-2
def optimize_prompt(prompt):
    if not prompt or prompt.strip() == "":
        return "Please enter a prompt to optimize."

    # Use GPT-2 to generate an optimized version of the prompt
    optimized_prompt = prompt.strip()
    base_prompt = f"Optimize this LLM prompt: '{optimized_prompt}'. Suggest a clearer, more specific version (e.g., add context, length, audience, or format): "

    # Generate a suggestion using GPT-2
    result = generator(base_prompt, num_return_sequences=1, max_length=100)[0]['generated_text']
    
    # Extract the optimized prompt (simplified parsing—improve later if needed)
    # Look for the suggestion after "Suggest a clearer, more specific version:"
    suggestion_start = result.find("Suggest a clearer, more specific version:") + len("Suggest a clearer, more specific version:")
    if suggestion_start > len("Suggest a clearer, more specific version:"):
        optimized_suggestion = result[suggestion_start:].strip()
    else:
        optimized_suggestion = optimized_prompt + " (add context or clarify intent, e.g., 'for a technical audience' or 'in 50 words')"

    return optimized_suggestion

# Route for the homepage (serves the HTML form)
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_prompt = request.form.get('prompt')
        optimized = optimize_prompt(user_prompt)
        return render_template('index.html', prompt=user_prompt, optimized=optimized)
    return render_template('index.html')

# API route for testing (optional, for future frontend integration)
@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided"}), 400
    optimized = optimize_prompt(data['prompt'])
    return jsonify({"original": data['prompt'], "optimized": optimized})

if __name__ == '__main__':
    app.run(debug=True)