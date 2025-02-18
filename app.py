import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

#loading the files and model
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model("chatbot_model.h5")

print(f"Loaded Words: {words}")
print(f"Loaded Classes: {classes}")

#clean the sentence function
def clean_sentence(sentence): 
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

#binanry vector of words 
def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    print(f"Bag Of Words: {bag}")
    return np.array(bag)

#predict the user sentence 'class'
def predict_class(sentence): 
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    print(f"prediction: {res}")
    error_threshold = 0.2

    results = [[i,r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key = lambda x: x[1], reverse = True)

    return_list = []
    for r in results:
        return_list.append({'intent' : classes[r[0]], 'probability' : str(r[1])})
        print(f"Results: {return_list}")
    return return_list

#chose a random response
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print('Working')

while True:
    message = input("")
    ints = predict_class (message)
    res = get_response(ints, intents)
    print(res)