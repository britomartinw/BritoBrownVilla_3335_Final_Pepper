from flask import Flask, request, jsonify, send_from_directory
from QASystem import preprocessText

import pickle
import json
import logging
from nltk.tokenize import word_tokenize

# Ensure NLTK data is downloaded if needed
# nltk.download('punkt')

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Flask app
app = Flask(__name__)

# Load the model, vectorizer, classes, and intents
try:
    with open('models/QAModel.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('models/TFIDFVectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    with open('models/IntentClasses.pkl', 'rb') as file:
        classes = pickle.load(file)

    with open('json/intents.json', 'r') as f:
        intents_data = json.load(f)

    logger.info("Successfully loaded model, vectorizer, classes, and intents.")

except Exception as e:
    logger.exception("Failed to load model files:")
    raise  # Critical error — stop the server if files are missing

# Helper function to find the response and media for a given predicted intent
def get_response_for_intent(intent_tag):
    """Return both answer and media for a predicted intent."""
    logger.info(f"Searching for intent: {intent_tag}")
    for intent in intents_data:
        logger.info(f"Checking intent: {intent.get('tag')}")
        if intent.get('tag') == intent_tag:
            answer = intent.get('answer', "Sorry, no response found.")
            media = intent.get('media', None)
            logger.info(f"Found intent. Answer: {answer}, Media: {media}")
            return answer, media
    logger.warning(f"No matching intent found for: {intent_tag}")
    return "Sorry, I don't have a response for that.", None

# Route to serve static media files (like images or videos)
@app.route('/media/<path:filename>')
def media(filename):
    return send_from_directory('media', filename)

# Prediction endpoint to handle POST requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate the input format
        if not data or 'input' not in data:
            return jsonify({'error': 'Invalid input format.'}), 400

        user_input = data['input']

        # Preprocess the input text (tokenize, normalize, etc.)
        tokens = preprocessText(user_input)
        preprocessed_input = ' '.join(tokens)

        # Transform the input using the TF-IDF vectorizer
        x = vectorizer.transform([preprocessed_input])

        # Predict the intent index and probabilities
        prediction_index = model.predict(x)[0]
        probabilities = model.predict_proba(x)[0]

        # Map prediction index to intent class
        predicted_intent = classes[prediction_index]
        confidence = float(probabilities[prediction_index]) if prediction_index < len(probabilities) else 0.0

        # Get answer and media for the predicted intent
        response_message, media = get_response_for_intent(predicted_intent)

        # Build the response dictionary
        response_data = {
            'intent': predicted_intent,
            'confidence': confidence,
            'response': response_message
        }

        # Include media if available
        if media:
            response_data['media'] = media

        logger.info("Sending response: %s", response_data)
        return jsonify(response_data)

    except Exception as e:
        logger.exception("Error during prediction:")
        return jsonify({'error': 'Internal server error'}), 500

# Endpoint to return the vocabulary used (for updating speech recognition, etc.)
@app.route('/vocabulary', methods=['GET'])
def get_vocabulary():
    try:
        logger.info("Fetching vocabulary: Starting extraction from intents data.")

        vocabulary = extract_keywords_from_intents(intents_data)

        logger.info("Vocabulary successfully extracted. Number of words: %d", len(vocabulary))
        logger.debug("Extracted vocabulary content: %s", vocabulary)

        logger.info("Sending vocabulary response to client: %s", json.dumps({"vocabulary": vocabulary}))

        return jsonify({"vocabulary": vocabulary})
    except Exception as e:
        logger.exception("Error fetching vocabulary:")
        return jsonify({'error': 'Internal server error'}), 500

# Helper function to extract all keywords from the intents
def extract_keywords_from_intents(intents_data):
    """Extract keywords and keyphrases properly from intents data."""
    keywords_set = set()

    if not intents_data:
        return []

    for intent in intents_data:
        keywords = intent.get('keywords', [])
        for keyword in keywords:
            clean_keyword = keyword.strip().lower()
            if clean_keyword:
                keywords_set.add(clean_keyword)

    return sorted(list(keywords_set))

# Main entry point for running the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
