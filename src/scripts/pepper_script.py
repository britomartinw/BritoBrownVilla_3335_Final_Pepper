import qi
import requests
import json
import time

class MyClass(GeneratedClass):
    def __init__(self):
        GeneratedClass.__init__(self, False)
        try:
            self.memory = ALProxy("ALMemory")  
            self.speech_recognition = ALProxy("ALSpeechRecognition")  
            self.tts = ALProxy("ALTextToSpeech")
            self.tablet_service = ALProxy("ALTabletService")
        except Exception as e:
            self.logger.error("Error in initialization: %s", str(e))
    
        # Centralize the Flask server address
        self.server_ip = "http://10.2.131.89:5000"
    
        # Initialize internal control flags
        self.bIsRunning = False
        self.hasSubscribed = False
        self.hasPushed = False
    
    

    def onLoad(self):
        self.logger.info("Pepper Speech Recognition Initialized")
        self.start_speech_recognition()

    def onUnload(self):
        self.logger.info("Pepper Speech Recognition Unloaded")
        self.stop_speech_recognition()

    def start_speech_recognition(self):
        """Start speech recognition by subscribing to the event and setting the vocabulary."""
        try:
            vocabulary = self.fetch_vocabulary_from_flask()
    
            if vocabulary:
                if isinstance(vocabulary, list) and all(isinstance(word, str) for word in vocabulary):
                    self.speech_recognition.pause(True)
                    self.logger.info("ASR engine paused.")
    
                    self.speech_recognition.removeAllContext()
                    self.logger.info("Existing grammar cleared.")
    
                    self.speech_recognition.setVocabulary(vocabulary, True)
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
        """Forcefully stop speech recognition and clear vocabulary safely."""
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
        self.logger.info("Word recognized event: %s", str(value))
        if len(value) > 1:
            raw_word = value[0]
            asr_confidence = value[1]
            confidence_threshold = 0.4
    
            # Clean recognized text due to word spotting
            recognized_word = raw_word.replace("<...>", "").strip()
            self.logger.info("Cleaned recognized word: %s", recognized_word)
    
            if recognized_word and asr_confidence >= confidence_threshold:
                self.logger.info("ASR confidence %.2f is good, sending to server.", asr_confidence)
                self.process_recognized_word(recognized_word)
            else:
                self.logger.info("Low ASR confidence (%.2f) or empty recognized text, asking for repeat.", asr_confidence)
                try:
                    self.speech_recognition.pause(True)
                    self.logger.info("ASR paused before speaking (low confidence).")
                    self.tts.say("I'm not sure I understood. Could you please repeat?")
                except Exception as e:
                    self.logger.error("Error during low-confidence TTS: %s", str(e))
                finally:
                    try:
                        self.speech_recognition.pause(False)
                        self.logger.info("ASR resumed after low-confidence handling.")
                    except Exception as e:
                        self.logger.error("Error resuming ASR: %s", str(e))
    
    
    def process_recognized_word(self, recognized_word):
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
                media = response_json.get('media', None)
    
                self.speech_recognition.pause(True)
                self.logger.info("ASR paused before speaking/displaying.")
    
                if response_message:
                    self.tts.say(str(response_message))
                    self.logger.info("Response spoken: %s", str(response_message))
    
                # Handle media if present
                if media and isinstance(media, dict):
                    media_type = media.get('type')
                    media_path = media.get('url')
    
                    if media_type and media_path:
                        full_media_url = self.server_ip + str(media_path) if media_path.startswith('/') else media_path
                        self.logger.info("Full media URL: %s", full_media_url)
    
                        try:
                            if media_type == 'image':
                                self.logger.info("Displaying image...")
                                self.tablet_service.showImage(full_media_url)
                                time.sleep(5)
                                self.tablet_service.hideImage()
                                self.logger.info("Image hidden after 5 seconds.")
    
                            elif media_type == 'video':
                                self.logger.info("Playing video...")
                                self.tablet_service.playVideo(full_media_url)
    
                                # You could also subscribe to VideoFinished for cleaner resuming
                                time.sleep(65)
                                self.logger.info("Video assumed complete after 65s.")
                            else:
                                self.logger.warning("Unsupported media type: %s", media_type)
    
                        except Exception as e:
                            self.logger.error("Error displaying media: %s", str(e))
                    else:
                        self.logger.warning("Media block missing 'type' or 'url'.")
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
                self.logger.info("ASR resumed after speaking/displaying.")
            except Exception as e:
                self.logger.error("Error resuming ASR: %s", str(e))
    
    
    
    def update_vocabulary_from_server(self):
        """Update the vocabulary dynamically from the Flask server."""
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
        """Send the recognized word/phrase to Flask server and return the response."""
        try:
            flask_url = self.server_ip + "/predict"
            payload = {'input': recognized_word}
            headers = {'Content-Type': 'application/json'}
    
            self.logger.info("Sending POST request to Flask server: %s with payload: %s", flask_url, str(payload))
    
            response = requests.post(flask_url, json=payload, headers=headers, timeout=10)
    
            self.logger.info("Raw HTTP response status: %d", response.status_code)
            self.logger.info("Raw HTTP response body: %s", str(response.text))
    
            if response.status_code == 200:
                response_json = response.json()
                self.logger.info("Received prediction response: %s", str(response_json))
                return response_json
            else:
                self.logger.error("Error from Flask server: Status code %d", response.status_code)
                return None
        except Exception as e:
            self.logger.error("Error sending request to Flask server: %s", str(e))
            return None
    

    def fetch_vocabulary_from_flask(self):
        """Fetch the dynamic vocabulary list from the Flask server."""
        try:
            flask_url = self.server_ip + "/vocabulary"
            self.logger.info("Fetching vocabulary from Flask server: %s", flask_url)
    
            response = requests.get(flask_url, timeout=10)
    
            self.logger.info("Raw HTTP response status: %d", response.status_code)
            self.logger.info("Raw HTTP response body: %s", str(response.text))
    
            if response.status_code == 200:
                response_json = response.json()
                self.logger.info("Received response JSON: %s", str(response_json))
    
                vocabulary = response_json.get('vocabulary')
    
                if vocabulary:
                    # Force every word to become a regular str (not unicode)
                    vocabulary = [str(word) for word in vocabulary]
    
                    if isinstance(vocabulary, list) and all(isinstance(word, str) for word in vocabulary):
                        self.logger.info("Received valid vocabulary: %s", str(vocabulary))
                        return vocabulary
                    else:
                        self.logger.error("Vocabulary list after conversion is invalid.")
                        return None
                else:
                    self.logger.error("Vocabulary missing in server response.")
                    return None
            else:
                self.logger.error("Failed to fetch vocabulary: %s", response.status_code)
                return None
        except Exception as e:
            self.logger.error("Error fetching vocabulary from Flask server: %s", str(e))
            return None
            
    def onInput_onStop(self):
        """When STOP button is pressed, clean up ASR and TTS."""
        self.logger.info("STOP pressed — Cleaning up speech recognition and TTS...")
        self.stop_speech_recognition()
        self.onStopped()
        