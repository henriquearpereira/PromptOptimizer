// Define the API base URL
const API_BASE_URL = 'http://localhost:5001/api';

document.addEventListener('DOMContentLoaded', function() {
    // Get form elements
    const optimizeForm = document.getElementById('optimizeForm');
    const feedbackForm = document.getElementById('feedbackForm');
    const submitFeedbackButton = document.getElementById('submitFeedback');
    const resultsSection = document.getElementById('results');
    
    // Form submission for prompt optimization
    optimizeForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Get form values
        const prompt = document.getElementById('prompt').value;
        const detailLevel = document.getElementById('detail_level').value;
        const creativity = parseFloat(document.getElementById('creativity').value);
        
        // Validate prompt
        if (!prompt.trim()) {
            alert('Please enter a prompt to optimize.');
            return;
        }
        
        try {
            // Call the optimize API
            const response = await fetch(`${API_BASE_URL}/optimize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prompt: prompt,
                    detail_level: detailLevel,
                    creativity: creativity
                })
            });
            
            if (!response.ok) {
                throw new Error('API request failed');
            }
            
            const data = await response.json();
            
            // Parse the optimized response
            const optimizedText = parseOptimizedResponse(data.optimized);
            
            // Display results
            document.getElementById('originalPrompt').textContent = data.original;
            document.getElementById('optimizedPrompt').innerHTML = optimizedText;
            resultsSection.style.display = 'block';
            
            // Store the optimized prompt for feedback
            feedbackForm.dataset.prompt = data.optimized;
            
        } catch (error) {
            console.error('Error optimizing prompt:', error);
            alert('Error optimizing prompt. Please try again.');
        }
    });
    
    // Parse the optimized response into formatted HTML
    function parseOptimizedResponse(response) {
        // Check if response contains the expected format
        if (response.includes('Optimized prompt:') && response.includes('Supporting details:')) {
            // Extract the main optimized prompt
            const promptRegex = /Optimized prompt:(.*?)(?=\n\nSupporting details:)/s;
            const promptMatch = response.match(promptRegex);
            const optimizedPrompt = promptMatch ? promptMatch[1].trim() : '';
            
            // Extract supporting details
            const detailsRegex = /Supporting details:([\s\S]*)/;
            const detailsMatch = response.match(detailsRegex);
            let supportingDetails = '';
            
            if (detailsMatch) {
                // Convert bullet points to formatted HTML
                supportingDetails = detailsMatch[1]
                    .split('\n')
                    .filter(line => line.trim())
                    .map(line => {
                        // Check if line starts with a bullet point
                        if (line.trim().startsWith('-')) {
                            const [label, value] = line.split(':').map(part => part.trim());
                            if (value) {
                                return `<div class="detail-item">
                                    <span class="detail-label">${label.replace('-', '').trim()}:</span>
                                    <span class="detail-value">${value}</span>
                                </div>`;
                            }
                        }
                        return '';
                    })
                    .join('');
            }
            
            // Build the formatted HTML
            return `
                <div class="optimized-result">
                    <div class="optimized-prompt">${optimizedPrompt}</div>
                    <div class="supporting-details">
                        <h4>Supporting Details:</h4>
                        <div class="details-container">
                            ${supportingDetails}
                        </div>
                    </div>
                </div>
            `;
        }
        
        // If not in expected format, return as is
        return response;
    }
    
    // Submit feedback
    submitFeedbackButton.addEventListener('click', async function() {
        const rating = document.getElementById('rating').value;
        const comment = document.getElementById('comment').value;
        const prompt = feedbackForm.dataset.prompt;
        
        if (!prompt) {
            alert('No optimized prompt to provide feedback for.');
            return;
        }
        
        try {
            // Call the feedback API
            const response = await fetch(`${API_BASE_URL}/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prompt: prompt,
                    rating: rating,
                    comment: comment
                })
            });
            
            if (!response.ok) {
                throw new Error('API request failed');
            }
            
            const data = await response.json();
            alert('Thank you for your feedback!');
            
            // Reset feedback form
            document.getElementById('comment').value = '';
            
        } catch (error) {
            console.error('Error submitting feedback:', error);
            alert('Error submitting feedback. Please try again.');
        }
    });
});