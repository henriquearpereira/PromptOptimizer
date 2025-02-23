# app/main.py
from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__, static_folder='../static') 

# Load a pre-trained GPT-2 model for text generation
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad_token to eos_token (50256 for gpt2)
tokenizer.pad_token = tokenizer.eos_token

# Create a pipeline for text generation
generator = pipeline("text-generation", 
                    model=model, 
                    tokenizer=tokenizer, 
                    max_length=100, 
                    truncation=True, 
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id)

# Function to optimize prompts using GPT-2
def optimize_prompt(prompt):
    if not prompt or prompt.strip() == "":
        return "Please enter a prompt to optimize."

    # Use GPT-2 to generate an optimized version of the prompt
    optimized_prompt = prompt.strip()
    base_prompt = f"Optimize this LLM prompt: '{optimized_prompt}'. Suggest a clearer, more specific version (e.g., add context like audience, length, format, or purpose) and return only the optimized prompt, no extra text: "

    # Generate a suggestion using GPT-2
    try:
        result = generator(base_prompt, num_return_sequences=1, max_length=150)[0]['generated_text']
        
        # Clean up the result—look for the optimized prompt after the base prompt
        optimized_start = result.find(optimized_prompt) + len(optimized_prompt)
        if optimized_start > len(optimized_prompt):
            # Try to extract the suggestion after the original prompt
            suggestion = result[optimized_start:].strip()
            # Remove any trailing or leading noise (e.g., extra text, punctuation)
            if suggestion and suggestion[0] in [',', '.', ':', ' ']:
                suggestion = suggestion[1:].strip()
            if suggestion and suggestion[-1] in [',', '.', ':']:
                suggestion = suggestion[:-1].strip()
            if suggestion and "suggest" in suggestion.lower():
                suggestion = suggestion.split("suggest", 1)[1].strip()
            if suggestion:
                return optimized_prompt + " " + suggestion
        # Fallback if parsing fails
        return optimized_prompt + " (enhance with context, e.g., specify audience, length like 'under 50 words', or format)"
    except Exception as e:
        print(f"Error generating prompt: {e}")
        return optimized_prompt + " (enhance with context, e.g., specify audience, length like 'under 50 words', or format)"

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