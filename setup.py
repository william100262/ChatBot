import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

#Creating lists
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

#Creating a nested for loop to fill out the words, classes, and documments lists appropriately
for intent in intents['intents'] :
    for pattern in intent['patterns'] :
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#Lemmatizing each word in the words list and storing it back in the words list, we then sort the list
lemmatized_words = []

for word in words:
    if word not in ignore_letters:
        lemmatized_words.append(lemmatizer.lemmatize(word.lower()))

words = lemmatized_words
words = sorted(set(words))
classes = sorted(set(classes))

#serializing the words and classes list for time efficency 
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

print(f"Words after processing: {words}")

training = []
output_empty = [0] * len(classes)

#creating a bag of words
for document in documents:
    bag_of_words = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag_of_words.append(1) if word in word_patterns else bag_of_words.append(0)

    #Appending the bag_of_words List and output_row List to the training list
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag_of_words + output_row)

#Shuffling the training data so no patterns are found
random.shuffle(training)
training = np.array(training)

#extracing the input and output
train_x = training[:, :len(words)]
train_y = training[:, len(words):]

#Creating a neural network
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(32, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation = 'softmax'))

#optimizer for the neural network
sgd = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs = 200, batch_size = 5, verbose = 1, validation_split=0.2)

#saving the model
model.save('chatbot_model.h5')
print('Executed')