# Overview
This is a basic chatbot that is built using Python and Keras, that utilizies a basic nureal network model to classify user inputs and respons accurately. 
I created this project to help me understand how a neural network works and explore the basics of machine learning

## Features
. Implements a bag-of-words model 
. Includes a samll set of predefined responses
. Allows interaction via the terminal

## What I used
1. Python
2. Keras and Tenserflow (Neural network implementation)
3. NLTK (Natural Language Toolkit)
4. NumPy (Array operations)
5. Pickle (Object serialization)
6. JSON (Storing intents and responses)

## Setup
You need Python 3.x installed along with: 
pip install nltk tensorflow keras numpy pickle-mixin

You need to train the model first: 
python setup.py

Lastly you run the chatbot: 
python app.py

Type in the termianl for a response

## Limitations
. This is a basic chatbot so the respones are very limited
. The model can not handle complex conversations
. The model is fixed which means it will not learn overtime based on your responses
