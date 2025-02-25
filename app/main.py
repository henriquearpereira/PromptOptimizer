from flask import Flask, request, jsonify, render_template
import torch
import logging
import re
import os
from models.model_loader import get_tokenizer, get_model

# Configure logging
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__, static_folder='../static')

# Load tokenizer and model
tokenizer = get_tokenizer()
model = get_model()

if tokenizer is None or model is None:
    logging.error("Tokenizer or model failed to load. The application may not function correctly.")

def optimize_prompt(prompt, detail_level='general', creativity=0.7):
    logging.debug(f"Starting optimize_prompt with prompt: '{prompt}', detail_level: '{detail_level}', creativity: {creativity}")

    if not prompt or prompt.strip() == "":
        logging.debug("Prompt is empty or whitespace.")
        return "Please enter a prompt to optimize."

    optimized_prompt = prompt.strip()
    
    # Adjust base_prompt based on detail_level
    if detail_level == 'general':
        base_prompt = (
            "You are a prompt optimization expert. For this writing prompt:\n"
            f"'{optimized_prompt}'\n\n"
            "Provide a general improvement to the prompt without over-specifying.\n\n"
            "Respond in this exact format (replace text in brackets):\n"
            "Optimized prompt: [brief improved version]\n\n"
            "Supporting details:\n"
            "- Target audience: [audience]\n"
            "- Length: [suggestion]\n"
            "- Style: [approach]\n"
            "- Elements: [key components]\n"
            "Do not include any additional content beyond the optimized prompt and supporting details."
        )
    elif detail_level == 'detailed':
        base_prompt = (
            "You are a prompt optimization expert. For this writing prompt:\n"
            f"'{optimized_prompt}'\n\n"
            "Provide a detailed improvement to the prompt, including specific suggestions and examples.\n\n"
            "Respond in this exact format (replace text in brackets):\n"
            "Optimized prompt: [brief improved version]\n\n"
            "Supporting details:\n"
            "- Target audience: [audience]\n"
            "- Length: [suggestion]\n"
            "- Style: [approach]\n"
            "- Elements: [key components]\n"
            "- Example: [detailed example or additional guidance]\n\n"
            "Make sure the optimized prompt is creative, specific, and actionable.\n"
            "Do not include any additional content beyond the optimized prompt and supporting details."
        )
    elif detail_level == 'highly_specific':
        base_prompt = (
            "You are a prompt optimization expert. For this writing prompt:\n"
            f"'{optimized_prompt}'\n\n"
            "Provide a highly specific improvement to the prompt, including detailed examples and step-by-step guidance.\n\n"
            "Respond in this exact format (replace text in brackets):\n"
            "Optimized prompt: [brief improved version]\n\n"
            "Supporting details:\n"
            "- Target audience: [audience]\n"
            "- Length: [suggestion]\n"
            "- Style: [approach]\n"
            "- Elements: [key components]\n"
            "- Example: [detailed example or additional guidance]\n\n"
            "Ensure the optimized prompt is highly specific and includes actionable steps.\n"
            "Do not include any additional content beyond the optimized prompt and supporting details."
        )
    else:
        return "Invalid detail level. Please choose 'general', 'detailed', or 'highly_specific'."
    
    if tokenizer is None or model is None:
        return "Model not loaded. Please restart the app."
    
    logging.debug(f"Base prompt: '{base_prompt}'")

    try:
        inputs = tokenizer(base_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,  # Increased to allow for more detailed responses
            temperature=creativity,  # Adjust based on user preference
            top_k=50,
            top_p=0.95,  # Higher for more diverse outputs
            do_sample=True
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Enhanced cleanup
        result = result.replace(base_prompt, "").strip()
        # Remove various markers and unwanted content
        unwanted_patterns = [
            r'Response:', r'## INPUT ##', r'## OUTPUT ##',
            r'<\|endofgeneration\|>', r'<\|.*?\|>',
            r'\[your improved prompt\]', r'\[list the details\]',
            r'<\|question_end\|>', r'Solution:', 
            r'Question \d+:.*?(?=Optimized prompt:)',
            r'Original passage:.*?(?=Optimized prompt:)',
            r'Corrected passage:.*?(?=Optimized prompt:)',
            r'Follow-up exercises?:.*',  # Remove follow-up exercises
            r'How would you adapt.*',  # Remove adaptation suggestions
            r'Optimized prompt for.*',  # Remove unrelated prompts
            r'Supporting details for.*',  # Remove unrelated details
            r'Textbook.*',  # Remove textbook-like explanations
            r'Writing a poem can be.*',  # Remove unrelated explanations
            r'As an optimization expert.*'  # Remove self-referential text
        ]
        
        for pattern in unwanted_patterns:
            result = re.sub(pattern, '', result, flags=re.DOTALL)
        
        # Clean up extra whitespace while preserving structure
        result = re.sub(r'\s{3,}', '\n\n', result)
        result = result.strip()

        # Validate the structure of the response
        if not result.startswith("Optimized prompt:"):
            # Fallback: If the model doesn't follow the format, generate a default response
            result = (
                f"Optimized prompt: {optimized_prompt}\n\n"
                "Supporting details:\n"
                "- Target audience: General\n"
                "- Length: Flexible\n"
                "- Style: Neutral\n"
                "- Elements: Add more specificity\n"
                "- Example: N/A"
            )

        return result

    except Exception as e:
        logging.error(f"Error generating prompt: {e}")
        return "An error occurred while optimizing the prompt. Please try again."
    
@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Home route for the web app. Handles both GET and POST requests.
    """
    logging.debug("Home route accessed")
    if request.method == 'POST':
        user_prompt = request.form.get('prompt')
        detail_level = request.form.get('detail_level', 'general')
        creativity = float(request.form.get('creativity', 0.7))
        logging.debug(f"User prompt: {user_prompt}, detail_level: {detail_level}, creativity: {creativity}")
        optimized = optimize_prompt(user_prompt, detail_level, creativity)
        logging.debug(f"Optimized prompt: {optimized}")
        return render_template('index.html', prompt=user_prompt, optimized=optimized)
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    """
    API endpoint for optimizing prompts. Accepts JSON input.
    """
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided"}), 400
    
    # Get user preferences (default to 'general' and creativity=0.7)
    detail_level = data.get('detail_level', 'general')
    creativity = float(data.get('creativity', 0.7))
    
    optimized = optimize_prompt(data['prompt'], detail_level, creativity)
    return jsonify({"original": data['prompt'], "optimized": optimized})

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    API endpoint for receiving user feedback on optimized prompts.
    """
    data = request.json
    if not data or 'prompt' not in data or 'rating' not in data:
        return jsonify({"error": "No prompt or rating provided"}), 400
    
    # Log feedback (you can store this in a database later)
    logging.info(f"User feedback for prompt '{data['prompt']}': Rating={data['rating']}, Comment={data.get('comment', '')}")
    
    return jsonify({"status": "Feedback received"})

if __name__ == '__main__':
    app.run(debug=False, port=5001)