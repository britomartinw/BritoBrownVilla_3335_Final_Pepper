import json
import random
import re
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet = True)
nltk.download('stopwords', quiet = True)
nltk.download('wordnet', quiet = True)

FILEPATH = 'json\intents.json'
MODELPATH = 'models/QAModel.pkl'
VECTORIZERPATH = 'models/TFIDFVectorizer.pkl' 
CLASSPATH ='models/IntentClasses.pkl'

lemmatizer = WordNetLemmatizer()

def loadIntents():
    """
    load the intents data from the json file
    
    args:
        filePath (str): Path to the intents json file
    returns:
        dict:the loaded intents data
    """
    try: 
        with open(FILEPATH, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: file {FILEPATH} was not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: file {FILEPATH} is not a valid JSON file")

def preprocessText(text):
    """
    preprocess the input text by converting to lowercase, removing punctuation,
    token izing, removing stopwords, and lemmatizing
    
    args:
        text (str): input text to preprocess
    returns:
        list: list of preprocessed tokens
    """

    text = text.lower() # convert to lowercase

    text = re.sub(r'[^\w\s]', '', text) # remove punctuation and special characters

    tokens = word_tokenize(text) # tokenize the text

    stopWords = set(stopwords.words('english')) # stopwords set bc of course python already has them

    # remove stopwords that might be important for educational questions
    domainRelevant = {'how', 'what', 'where', 'when', 'why', 'who', 'which'}
    stopWords = stopWords - domainRelevant
    tokens = [word for word in tokens if word not in stopWords]

    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def extractFeatures(intentsData):
    """
    extract features and prepare training data from the intents data
    
    args:
        intentsData (dict): the loaded intents data
    returns:
        tuple: (xTrain, yTrain, vectorizer, classes)
            xTrain - TF-IDF vectorized training data examples
            yTrain - labels for the trainin examples
            vectorizer - fitted TF-IDF vectorizer
            classes - list of intent classes 
    """

    trainingData = []
    classes = []
    documents = []

    # extract the patterns and their corresponding tags
    for intent in intentsData:
        tag = intent['question']
        if tag not in classes:
            classes.append(tag)

        for pattern in intent['patterns']:

            # preprocess the pattern
            tokens = preprocessText(pattern)
            preprocessedPattern = ' '.join(tokens)

            # add to documents
            documents.append((preprocessedPattern, tag))
            trainingData.append(preprocessedPattern)

    # create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features = 2000) # increased features for more precision
    # extract features using TF-IDF 
    xTrain = vectorizer.fit_transform(trainingData)
    # prepare labels
    yTrain = []
    for doc in documents:
        yTrain.append(classes.index(doc[1]))

    return xTrain, np.array(yTrain), vectorizer, classes 

def trainModel(xTrain, yTrain):
    """
    train a classification model on the given training data

    args:
        xTrain: vectorized training examples
        yTrain: labels for the training examples
    returns:
        model: trained classfication model 
    """

    # using logistic regression as our classification model with tuned parameters
    model = LogisticRegression(max_iter = 2000, C = 2.0, solver = 'lbfgs', multi_class = 'multinomial', class_weight = 'balanced')
    model.fit(xTrain, yTrain)

    # make prediction on the training data to evaluate
    yPred = model.predict(xTrain)

    # print model evaluation metrics
    print("Model training complete")
    print(f"Accuracy: {accuracy_score(yTrain, yPred):.4f}")
    print("\nClassification Report:")
    print(classification_report(yTrain, yPred))

    return model

def predictIntent(userInput, vectorizer, model, classes):
    """
    predict the intetn of the user's input message

    args:
        userInput (str): the user's input message
        vectorizer: TF-IDF vectorizer fitted on training data
        model: trained classification model
        classes (list): list of intent classes
    returns:
        tuple (predictedIntent, confidence)
            predictedIntent - the predicted intent tag
            confidence - confidence score for the prediction    
    """

    # preprocess the user input
    tokens = preprocessText(userInput)
    preprocessedInput = ' '.join(tokens)

    # vectorize the preprocessed input
    x = vectorizer.transform([preprocessedInput])

    # predict the intent
    prediction = model.predict(x)[0]

    # get the prediction probabilities
    probabilities = model.predict_proba(x)[0]
    confidence = probabilities[prediction]
    return classes[prediction], confidence

def getResponse(predictedIntent, confidence, intentsData, confidenceThreshold = 0.1):
    """
    get a response based on the predicted intent and confidence level

    args:
        predictedIntent (str): the predicted intent tag
        confidence (float): confidence score for the prediction
        intentsData (dcit): the loaded intents data
        confidenceThreshold (float): threshold for confidence to accept the prediction
    returns:
        tuple: (responseMessage, confidenceCategory)
            responseMessage - the answer or clarification request
            confidenceCategory - "high", "medium", or "low"
    """

    # define confidence levels
    if confidence < confidenceThreshold:
        # if we end up having a sheet where we list questions we might want to cater a response to make
        # suggest to the little fucker to use their eyes and read it
        return "I'm not sure i understand. Could you please rephrase or provide more details", "low"
    
    # find the corresponding intent and select a random response
    for intent in intentsData:
        if intent['question'] == predictedIntent:
            confidenceCategory = "high" if confidence > 0.7 else "medium"
            return intent['answer'], confidenceCategory
        
    # fallback response if intetnt wasn't found (shouldn't happen)
    return "I apologize, I'm having trouble understanding, please try again", "low"

def saveModel(model, vectorizer, classes):
    """
    save the trained model, vectorizer, and classes to files

    args:
        model - trained classification model
        vectorizer - fitted TF-IDF vectorizer
        classes (list) - list of intent classes
        modelPath (str) - path to save the model
        vectorizerPath (str) - path to save the vectorizer
        classesPath (str) - path to save the intent classes 
    """

    with open(MODELPATH, 'wb') as file:
        pickle.dump(model, file)

    with open(VECTORIZERPATH, 'wb') as file:
        pickle.dump(vectorizer, file)

    with open(CLASSPATH, 'wb') as file:
        pickle.dump(classes, file)

def loadModel(modelPath = "QAModel.pkl", vectorizerPath = "TFIDFVectorizer.pkl", classesPath = 'IntentClasses.pkl'):
    """
    load the trained model, vectorizer, and classes from files

    args:
        modelPath (str) - path to the saved model
        vectorizerPath (str) - path to the saved vectorizer
        classesPath (str) - path to the saved intent classes
    returns:
        tuple: (model, vectorizer, classes)
            model - loaded classification model
            vectorizer - loaded TF-IDF vectorizer
            classes - list of intent classes 
    """

    try:
        with open(MODELPATH, 'rb') as file:
            model = pickle.load(file)

        with open(VECTORIZERPATH, 'rb') as file:
            vectorizer = pickle.load(file)

        with open(CLASSPATH, 'rb') as file:
            classes = pickle.load(file)

        print("Model, vectorizer, and classes loaded successfully")
        return model, vectorizer, classes
    except (FileNotFoundError, EOFError, ValueError) as e:
        print(f"Error loading model files: {e}")
        print("Model files not found or corrupted, training a new model...")
        return None, None, None
    
def runQASystem():
    """
    run the QA system interface fpr demonstration
    """

    print("=" * 50)
    print("QA System")
    print("=" * 50)
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("=" * 50)

    # try to load the trained model or train one if it isn't found
    model, vectorizer, classes = loadModel()
    if model is None:
        print("Model not found, training a new model...")

        # load intents data
        intentsData = loadIntents()
        if intentsData is None:
            print("Error loading intents data")
            return
        
        # extract features and train model
        xTrain, yTrain, vectorizer, classes = extractFeatures(intentsData)
        model = trainModel(xTrain, yTrain)

        # save the model for future use
        saveModel(model, vectorizer, classes)


    # load intents data for responses
    intentsData = loadIntents()
    if intentsData is None:
        print("Error loading intents data")
        return
    
    # main conversation loop
    while True:

        # get user input
        userInput = input("\nYou: ")

        # check if user wants to exit
        if userInput.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print("\nSystem: Bye byyyeee, have a beautiful tiiimee")
            break
        
        # predict intent and get response 
        predictedIntent, confidence = predictIntent(userInput, vectorizer, model, classes)
        response, confidenceLevel = getResponse(predictedIntent, confidence, intentsData)

        # print response 
        print(f"\nSystem: {response}")
        print(f"(Debug - intent: {predictedIntent}, Confidence: {confidence:.4f}, Level: {confidenceLevel})")

if __name__ == "__main__":
    runQASystem()