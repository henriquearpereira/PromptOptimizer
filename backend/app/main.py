from flask import Flask, request, jsonify
import torch
import logging
import re
import os
from flask_cors import CORS
from app.models.model_loader import get_tokenizer, get_model

# Configure logging
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend to communicate with backend

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

    original_prompt = prompt.strip()
    
    # Enhanced system prompt with more specific instructions
    system_prompt = (
        "You are PromptOptimizer, a specialized AI for enhancing writing prompts. "
        "You transform basic prompts into detailed, creative, and actionable instructions. "
        "Your response MUST follow the exact format specified."
    )
    
    # Adjust base_prompt based on detail_level with improved examples
    if detail_level == 'general':
        base_prompt = (
        f"{system_prompt}\n\n"
        f"ORIGINAL PROMPT: {original_prompt}\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the prompt's core intention.\n"
        "2. Add basic context and direction to make it more actionable.\n"
        "3. Maintain the original idea while enhancing its clarity.\n"
        "4. Provide an optimized prompt and general supporting details.\n\n"
        "EXAMPLE - if given 'Write a blog post':\n"
        "Optimized prompt: Write a blog post about the benefits of mindful meditation for beginners.\n\n"
        "Supporting details:\n"
        "- Target audience: Individuals new to meditation\n"
        "- Length: Approximately 500-700 words\n"
        "- Style: Informative and accessible\n"
        "- Elements: Clear explanations, practical tips, personal anecdotes (optional)\n\n"
        "BAD EXAMPLE - if given 'Write a blog post':\n"
        "Optimized prompt: Write a blog post.\n\n"
        "Supporting details:\n"
        "- Target audience: General\n"
        "- Length: Flexible\n"
        "- Style: Neutral\n"
        "- Elements: Write things\n"
        "Explanation of why the bad example is bad: This example does not add any value to the original prompt, and does not provide any supporting details that could help a writer."
    )
    elif detail_level == 'detailed':
        base_prompt = (
        f"{system_prompt}\n\n"
        f"ORIGINAL PROMPT: {original_prompt}\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the prompt thoroughly and add significant detail.\n"
        "2. Provide an optimized prompt that transforms the original with creative direction.\n"
        "3. List detailed supporting details, including specific style and element recommendations.\n\n"
        "EXAMPLE - if given 'Create a social media campaign':\n"
        "Optimized prompt: Create a social media campaign for a sustainable fashion brand, focusing on visually appealing content that highlights ethical sourcing and production.\n\n"
        "Supporting details:\n"
        "- Target audience: Environmentally conscious young adults (18-35)\n"
        "- Length: 3-5 posts per week for one month\n"
        "- Style: Visually driven, authentic, and engaging\n"
        "- Elements: Behind-the-scenes content, customer testimonials, educational infographics, interactive polls\n"
        "- Example: A series of Instagram stories showcasing the journey of a garment from raw material to finished product.\n\n"
        "BAD EXAMPLE - if given 'Create a social media campaign':\n"
        "Optimized prompt: Create a social media campaign.\n\n"
        "Supporting details:\n"
        "- Target audience: General\n"
        "- Length: Flexible\n"
        "- Style: Neutral\n"
        "- Elements: Use social media\n"
        "Explanation of why the bad example is bad: This example does not add any value to the original prompt, and does not provide any supporting details that could help a content creator."
    )
    elif detail_level == 'highly_specific':
        base_prompt = (
        f"{system_prompt}\n\n"
        f"ORIGINAL PROMPT: {original_prompt}\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the prompt's full potential and provide a comprehensive transformation.\n"
        "2. Provide an optimized prompt with specific parameters and actionable structure.\n"
        "3. List highly detailed supporting details, including precise length requirements and style guides.\n\n"
        "EXAMPLE - if given 'Design a website':\n"
        "Optimized prompt: Design a responsive e-commerce website for a local artisan bakery, focusing on a clean, visually appealing layout that showcases product photography and facilitates easy online ordering.\n\n"
        "Supporting details:\n"
        "- Target audience: Local residents and tourists seeking high-quality baked goods (25-55 years old)\n"
        "- Length: 5-7 pages, including a homepage, product catalog, about us, contact, and checkout\n"
        "- Style: Minimalist, modern, and user-friendly with a warm color palette\n"
        "- Elements: High-resolution product photos, clear call-to-action buttons, customer reviews, secure payment gateway, mobile responsiveness\n"
        "- Example: A homepage featuring a rotating carousel of featured products, followed by a section highlighting customer favorites and a newsletter signup form.\n\n"
        "BAD EXAMPLE - if given 'Design a website':\n"
        "Optimized prompt: Design a website.\n\n"
        "Supporting details:\n"
        "- Target audience: General\n"
        "- Length: Flexible\n"
        "- Style: Neutral\n"
        "- Elements: Use web design\n"
        "Explanation of why the bad example is bad: This example does not add any value to the original prompt, and does not provide any supporting details that could help a web designer."
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
            max_new_tokens=350,  # Increased for more detailed responses
            temperature=creativity,  # Adjust based on user preference
            top_k=50,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response portion
        response_start = result.find("Optimized prompt:")
        if response_start == -1:
            # If the model didn't follow the format, extract what might be the response
            # Look for patterns that might indicate the start of the response
            possible_starts = [
                "Here's an optimized version:",
                "Improved prompt:",
                "Enhanced prompt:",
                "OPTIMIZED PROMPT:"
            ]
            
            for start_phrase in possible_starts:
                start_idx = result.find(start_phrase)
                if start_idx != -1:
                    response_start = start_idx
                    result = "Optimized prompt:" + result[start_idx + len(start_phrase):]
                    break
        
        if response_start == -1:
            # Still couldn't find a proper start, use the content after the example
            example_end = result.find("EXAMPLE")
            if example_end != -1:
                # Find the next paragraph after the example
                next_para = result.find("\n\n", example_end)
                if next_para != -1:
                    result = result[next_para:].strip()
                else:
                    # Fallback to default response
                    return generate_fallback_response(original_prompt, detail_level)
            else:
                # Fallback to default response
                return generate_fallback_response(original_prompt, detail_level)
        else:
            result = result[response_start:].strip()
            
        # Clean up the response
        # Remove any system messages that might have been included
        system_messages = [
            "I'll optimize this prompt for you.",
            "Here's how I would optimize this prompt:",
            "Let me enhance this prompt for you:",
            "Based on your request, I've optimized the prompt:"
        ]
        
        for msg in system_messages:
            if result.startswith(msg):
                result = result[len(msg):].strip()
                
        # Ensure we have the correct format
        if not result.startswith("Optimized prompt:"):
            result = "Optimized prompt: " + result
            
        # Make sure we have supporting details
        if "Supporting details:" not in result:
            result += "\n\nSupporting details:\n"
            result += "- Target audience: General\n"
            result += "- Length: Appropriate for content type\n"
            result += "- Style: Engaging and clear\n"
            result += "- Elements: Specific details, context, purpose"
            
        return result

    except Exception as e:
        logging.error(f"Error generating prompt: {e}")
        return "An error occurred while optimizing the prompt. Please try again."
        
def generate_fallback_response(original_prompt, detail_level):
    """Generate a fallback response that's better than the default generic one"""
    
    # Create more specific fallback responses based on prompt type
    prompt_lower = original_prompt.lower()
    
    # Detect common prompt types
    if "poem" in prompt_lower or "poetry" in prompt_lower:
        return (
            f"Optimized prompt: Write a reflective poem that uses vivid natural imagery to explore the theme of {get_theme_from_prompt(original_prompt) or 'personal transformation'}.\n\n"
            "Supporting details:\n"
            "- Target audience: Poetry readers interested in introspective works\n"
            "- Length: 14-20 lines arranged in 3-4 stanzas\n"
            "- Style: Free verse with attention to rhythm and sensory language\n"
            "- Elements: Nature metaphors, emotional journey, specific sensory details, surprising conclusion"
        )
    elif "story" in prompt_lower or "narrative" in prompt_lower or "fiction" in prompt_lower:
        return (
            f"Optimized prompt: Write a character-driven short story about a person who discovers {get_theme_from_prompt(original_prompt) or 'an unexpected truth'} that changes their perspective on life.\n\n"
            "Supporting details:\n"
            "- Target audience: Adult fiction readers who enjoy psychological depth\n"
            "- Length: 1500-2000 words with clear beginning, middle, and end\n"
            "- Style: Third-person limited perspective with sensory details and meaningful dialogue\n"
            "- Elements: Flawed protagonist, inciting incident, internal conflict, moment of realization, changed behavior"
        )
    elif "essay" in prompt_lower or "article" in prompt_lower or "research" in prompt_lower:
        return (
            f"Optimized prompt: Write an informative essay exploring how {get_theme_from_prompt(original_prompt) or 'modern technology'} is reshaping our understanding of {get_secondary_theme_from_prompt(original_prompt) or 'human connection'}.\n\n"
            "Supporting details:\n"
            "- Target audience: Educated general readers interested in thoughtful analysis\n"
            "- Length: 1200-1500 words with clear thesis and supporting evidence\n"
            "- Style: Balanced academic tone with accessible language and compelling examples\n"
            "- Elements: Current research, historical context, contrasting viewpoints, real-world implications, call to reflection"
        )
    else:
        # General fallback
        return (
            f"Optimized prompt: {enhance_general_prompt(original_prompt)}\n\n"
            "Supporting details:\n"
            "- Target audience: Interested readers seeking depth and insight\n"
            "- Length: Appropriate for the content type (approximately 800-1200 words)\n"
            "- Style: Clear, engaging, with a mix of analysis and illustrative examples\n"
            "- Elements: Specific focus, contextual background, thoughtful development, memorable takeaways"
        )

def get_theme_from_prompt(prompt):
    """Extract a possible theme from the original prompt"""
    # Look for theme indicators
    theme_indicators = [
        "about", "on", "regarding", "concerning", 
        "exploring", "discussing", "examining"
    ]
    
    for indicator in theme_indicators:
        indicator_phrase = f"{indicator} "
        if indicator_phrase in prompt.lower():
            idx = prompt.lower().find(indicator_phrase) + len(indicator_phrase)
            theme = prompt[idx:].strip()
            # If theme has punctuation, strip it
            theme = theme.split(".")[0].split("?")[0].split("!")[0].strip()
            return theme
    
    return None

def get_secondary_theme_from_prompt(prompt):
    """Extract a secondary theme or connect to a relevant concept"""
    # Common themes to suggest
    themes = ["personal growth", "social change", "ethical considerations", 
              "future implications", "historical lessons", "human connection"]
    
    # Simple heuristic - return one based on prompt length
    return themes[len(prompt) % len(themes)]

def enhance_general_prompt(prompt):
    """Add specificity to a general prompt"""
    prompt = prompt.strip()
    
    # If prompt is very short (less than 5 words)
    if len(prompt.split()) < 5:
        base = prompt.rstrip('.?!')
        enhancements = [
            f"{base} that explores the tension between tradition and innovation in contemporary society.",
            f"{base} with a focus on how small choices lead to significant life changes.",
            f"{base} examining the relationship between human identity and technological advancement.",
            f"{base} that challenges conventional wisdom about success and fulfillment."
        ]
        # Use prompt length to select an enhancement deterministically
        return enhancements[len(prompt) % len(enhancements)]
    else:
        # For longer prompts, add specificity about approach
        specifiers = [
            "using concrete examples and counterexamples",
            "through the lens of both historical precedent and future possibility",
            "with attention to both emotional and logical dimensions",
            "considering diverse perspectives and unexpected connections"
        ]
        return f"{prompt.rstrip('.?!')} {specifiers[len(prompt) % len(specifiers)]}."

@app.route('/api/optimize', methods=['POST'])
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

@app.route('/api/feedback', methods=['POST'])
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