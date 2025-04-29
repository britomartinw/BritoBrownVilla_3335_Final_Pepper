# Script for Pepper Robot: MerrimackOpenHouse Behavior
# Connects to a Flask server for dynamic speech recognition and media response.

import qi
import requests
import json
import time

class MyClass(GeneratedClass):
    def __init__(self):
        GeneratedClass.__init__(self, False)
        try:
            # Setup connections to Pepper services
            self.memory = ALProxy("ALMemory")  
            self.speech_recognition = ALProxy("ALSpeechRecognition")  
            self.tts = ALProxy("ALTextToSpeech")
            self.tablet_service = ALProxy("ALTabletService")
        except Exception as e:
            self.logger.error("Error in initialization: %s", str(e))
        
        # IP address of the Flask server
        self.server_ip = "http://10.2.131.89:5000"

        # Internal control flags
        self.bIsRunning = False
        self.hasSubscribed = False
        self.hasPushed = False

    def onLoad(self):
        """Called when behavior starts: initialize ASR (Automatic Speech Recognition)."""
        self.logger.info("Pepper Speech Recognition Initialized")
        self.start_speech_recognition()

    def onUnload(self):
        """Called when behavior stops: cleanup."""
        self.logger.info("Pepper Speech Recognition Unloaded")
        self.stop_speech_recognition()

    def start_speech_recognition(self):
        """Start ASR by setting the vocabulary fetched from the Flask server."""
        try:
            vocabulary = self.fetch_vocabulary_from_flask()
    
            if vocabulary:
                if isinstance(vocabulary, list) and all(isinstance(word, str) for word in vocabulary):
                    self.speech_recognition.pause(True)  # Pause ASR engine
                    self.logger.info("ASR engine paused.")
    
                    self.speech_recognition.removeAllContext()  # Clear any previous context
                    self.logger.info("Existing grammar cleared.")
    
                    self.speech_recognition.setVocabulary(vocabulary, True)  # Set new vocabulary
                    self.logger.info("Vocabulary set: %s", str(vocabulary))
    
                    self.memory.subscribeToEvent("WordRecognized", self.getName(), "onWordRecognized")
                    self.hasSubscribed = True
                    self.logger.info("Subscribed to WordRecognized event.")
                else:
                    self.logger.error("Invalid vocabulary format received from Flask server.")
            else:
                self.logger.error("No vocabulary received.")
        except Exception as e:
            self.logger.error("Error in start_speech_recognition: %s", str(e))
    
    def stop_speech_recognition(self):
        """Safely stop speech recognition and reset ASR."""
        try:
            if hasattr(self, "memory") and self.hasSubscribed:
                self.memory.unsubscribeToEvent("WordRecognized", self.getName())
                self.logger.info("Unsubscribed from WordRecognized event.")
                self.hasSubscribed = False
    
            if hasattr(self, "speech_recognition") and self.hasPushed:
                self.speech_recognition.popContexts()
                self.logger.info("ASR context popped.")
                self.hasPushed = False
    
            if hasattr(self, "speech_recognition") and self.speech_recognition:
                self.speech_recognition.removeAllContext()
                self.speech_recognition.setVocabulary([], False)
                self.speech_recognition.setVisualExpression(False)
                self.logger.info("Cleared ASR vocabulary.")
    
            if hasattr(self, "tts") and self.tts:
                self.tts.stopAll()
                self.logger.info("TTS speaking stopped.")
    
        except Exception as e:
            self.logger.error("Error stopping speech recognition: %s", str(e))
        finally:
            self.bIsRunning = False

    def onWordRecognized(self, key, value, message):
        """Triggered when a word is recognized by ASR."""
        self.logger.info("Word recognized event: %s", str(value))
        if len(value) > 1:
            raw_word = value[0]
            asr_confidence = value[1]
            confidence_threshold = 0.4  # Minimum confidence required

            # Clean recognized word
            recognized_word = raw_word.replace("<...>", "").strip()
            self.logger.info("Cleaned recognized word: %s", recognized_word)
    
            if recognized_word and asr_confidence >= confidence_threshold:
                self.logger.info("ASR confidence %.2f is good, sending to server.", asr_confidence)
                self.process_recognized_word(recognized_word)
            else:
                # If low confidence, ask user to repeat
                self.logger.info("Low ASR confidence (%.2f) or empty recognized text, asking for repeat.", asr_confidence)
                try:
                    self.speech_recognition.pause(True)
                    self.tts.say("I'm not sure I understood. Could you please repeat?")
                except Exception as e:
                    self.logger.error("Error during low-confidence TTS: %s", str(e))
                finally:
                    try:
                        self.speech_recognition.pause(False)
                    except Exception as e:
                        self.logger.error("Error resuming ASR: %s", str(e))
    
    def process_recognized_word(self, recognized_word):
        """Send the recognized word to the Flask server and handle the response."""
        if not recognized_word:
            return
    
        try:
            flask_url = self.server_ip + "/predict"
            payload = {'input': recognized_word}
            headers = {'Content-Type': 'application/json'}
    
            self.logger.info("Sending POST request to Flask server: %s", flask_url)
            response = requests.post(flask_url, json=payload, headers=headers, timeout=10)
    
            self.logger.info("Raw HTTP response status: %d", response.status_code)
            self.logger.info("Raw HTTP response body: %s", str(response.text))
    
            if response.status_code == 200:
                response_json = response.json()
                self.logger.info("Received prediction response: %s", str(response_json))
    
                response_message = response_json.get('response')
                media = response_json.get('media', None)  # Optional media field

                self.speech_recognition.pause(True)  # Pause ASR before playing response
    
                if response_message:
                    self.tts.say(str(response_message))
                    self.logger.info("Response spoken: %s", str(response_message))

                # Handle media (image or video)
                if media and isinstance(media, dict):
                    media_type = media.get('type')
                    media_url = media.get('url')
                
                    if media_type and media_url:
                        media_url = str(media_url)
                        try:
                            if media_type == 'image':
                                self.logger.info("Displaying image: %s", media_url)
                                self.tablet_service.showImage(media_url)
                                time.sleep(5)  # Show image for 5 seconds
                                self.tablet_service.hideImage()
                            elif media_type == 'video':
                                self.logger.info("Playing video: %s", media_url)
                                self.speech_recognition.pause(True)
                                self.tablet_service.playVideo(media_url)
                                time.sleep(65)  # Assume video duration
                                self.speech_recognition.pause(False)
                            else:
                                self.logger.warning("Unknown media type: %s", media_type)
                        except Exception as e:
                            self.logger.error("Error displaying media on tablet: %s", str(e))
                    else:
                        self.logger.warning("Media field missing 'type' or 'url'.")
                else:
                    self.logger.info("No media to display.")
    
            else:
                self.logger.error("Flask server error: %d", response.status_code)
                self.tts.say("Sorry, server error.")
    
        except Exception as e:
            self.logger.error("Error sending recognized word: %s", str(e))
            self.tts.say("Sorry, I couldn't reach the server.")
    
        finally:
            try:
                self.speech_recognition.pause(False)
            except Exception as e:
                self.logger.error("Error resuming ASR: %s", str(e))
    
    def update_vocabulary_from_server(self):
        """Manually update vocabulary from the Flask server."""
        try:
            vocabulary = self.fetch_vocabulary_from_flask()
            if vocabulary:
                if isinstance(vocabulary, list) and all(isinstance(word, str) for word in vocabulary):
                    self.speech_recognition.setVocabulary(vocabulary, False)
                    self.logger.info("Vocabulary updated dynamically.")
                else:
                    self.logger.error("Invalid vocabulary format received from Flask server.")
            else:
                self.logger.error("Failed to fetch vocabulary from Flask server.")
        except Exception as e:
            self.logger.error("Error updating vocabulary: %s", str(e))

    def send_to_flask_server(self, recognized_word):
        """Helper to send a recognized word to Flask and return server's prediction."""
        try:
            flask_url = self.server_ip + "/predict"
            payload = {'input': recognized_word}
            headers = {'Content-Type': 'application/json'}
    
            self.logger.info("Sending POST request to Flask server: %s", flask_url)
            response = requests.post(flask_url, json=payload, headers=headers, timeout=10)
    
            if response.status_code == 200:
                response_json = response.json()
                return response_json
            else:
                self.logger.error("Flask server returned error: %d", response.status_code)
                return None
        except Exception as e:
            self.logger.error("Error sending request to Flask server: %s", str(e))
            return None
    
    def fetch_vocabulary_from_flask(self):
        """Fetch the latest vocabulary from the Flask server."""
        try:
            flask_url = self.server_ip + "/vocabulary"
            self.logger.info("Fetching vocabulary from Flask server: %s", flask_url)
            response = requests.get(flask_url, timeout=10)
    
            if response.status_code == 200:
                response_json = response.json()
                vocabulary = response_json.get('vocabulary')
    
                if vocabulary:
                    vocabulary = [str(word) for word in vocabulary]
                    if isinstance(vocabulary, list) and all(isinstance(word, str) for word in vocabulary):
                        return vocabulary
                    else:
                        self.logger.error("Invalid vocabulary after conversion.")
                        return None
                else:
                    self.logger.error("Vocabulary field missing.")
                    return None
            else:
                self.logger.error("Flask server error: %d", response.status_code)
                return None
        except Exception as e:
            self.logger.error("Error fetching vocabulary: %s", str(e))
            return None

    def onInput_onStop(self):
        """Called when STOP button is pressed to safely clean up resources."""
        self.logger.info("STOP pressed — Cleaning up speech recognition and TTS...")
        self.stop_speech_recognition()
        self.onStopped()
