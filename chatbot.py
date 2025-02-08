import json
import numpy as np
import pickle
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load trained data
words = pickle.load(open("models/words.pkl", "rb"))
classes = pickle.load(open("models/classes.pkl", "rb"))
model = load_model("models/chatbot_model.h5")

with open("data/intents.json") as file:
    intents = json.load(file)

def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    return np.array([1 if w in sentence_words else 0 for w in words])

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    return [{"intent": classes[i], "probability": str(r)} for i, r in enumerate(res) if r > 0.25]

def get_response(intents_list):
    if not intents_list:
        return "I'm sorry, I don't understand. Can you rephrase your question?"
    
    tag = intents_list[0]["intent"]
    for i in intents["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])
    return "I'm sorry, I don't understand."

