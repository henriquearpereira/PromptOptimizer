# app/main.py
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Hardcoded prompt optimization function (replace with LLM logic later)
def optimize_prompt(prompt):
    if not prompt or prompt.strip() == "":
        return "Please enter a prompt to optimize."
    
    # Simple hardcoded optimizations based on prompt content
    optimized_prompt = prompt.strip()
    if "write" in optimized_prompt.lower():
        optimized_prompt += " (be specific: include length, style, or context, e.g., '100 words, poetic style')"
    elif "explain" in optimized_prompt.lower():
        optimized_prompt += " (provide examples or break it into steps for clarity)"
    elif "generate" in optimized_prompt.lower():
        optimized_prompt += " (specify format, e.g., 'list, paragraph, or code')"
    else:
        optimized_prompt += " (add context or clarify intent, e.g., 'for a technical audience' or 'in 50 words')"
    
    return optimized_prompt

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