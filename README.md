# CSC3335 Final Project
# Pepper Robot - Merrimack Open House Project

This project controls Pepper's behavior using a Flask server backend.  
Pepper dynamically listens for user speech, sends recognized phrases to a server for prediction, and responds with speech or media content (images/videos).

It combines **robotics**, **natural language processing (NLP)**, **machine learning (ML)**, **speech recognition**, and **database management** into a dynamic, intelligent system.

This project was designed for CSC3335 Artificial Intelligence Final Project

---

## Project Structure

- **Pepper Behavior Script**: Manages Pepper’s speech recognition, text-to-speech, and tablet display services.
- **Flask Server**: Receives recognized phrases, predicts intent using an NLP + ML model, and sends appropriate responses.
- **SQL Database**: Stores intents, responses, and media (images/videos) linked to each intent.
- **Tablet Media Support**: Displays images or videos on Pepper's tablet based on server predictions.
- **NLP + ML Model**: Built, trained, and serialized using Python libraries like NLTK, scikit-learn, and Pickle.

---

## How It Works

1. Pepper initializes and fetches a dynamic vocabulary list from the Flask server.
2. Pepper listens for user phrases through Automatic Speech Recognition (ASR).
3. On recognized speech:
   - Sends the phrase to the Flask server’s `/predict` endpoint.
   - The server uses a **pre-trained NLP + ML model** to predict the most appropriate intent.
   - The server queries the **MySQL database** for the corresponding response and optional media.
   - The server returns a response to Pepper.
4. Pepper speaks the text response, displays an image, or plays a video on its tablet.

---

## Installation & Setup

### Requirements on Pepper:
- Choregraphe installed (for behavior development and upload).
- NAOqi SDK available for Python scripting on Pepper.

### Server Requirements:
- Python 3.x
- Flask
- NLTK
- scikit-learn
- NumPy
- MySQL Database
- MySQL Connector
- Requests Library
- Gson

Install Python dependencies via:

```bash
pip install flask nltk scikit-learn numpy requests logging

```

## Contributors: 
- Wander Brito Martinez
- Michael Brown
- Nayeli Villa