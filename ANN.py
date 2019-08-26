# -*- coding: utf-8 -*-

#Part 1: NLP


# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('data_from_api_main.tsv', delimiter = '\t')

# Cleaning the texts using stemming
'''
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 5887):    
    question = re.sub('[^a-zA-Z]', ' ', dataset['Questions'][i])
    question = question.lower()
    question = question.split()
    ps = PorterStemmer()
    question = [ps.stem(word) for word in question if not word in set(stopwords.words('english'))]
    question = ' '.join(question)
    corpus.append(question)
'''

#Cleaning using Lemmatisation
#https://www.guru99.com/stemming-lemmatization-python-nltk.html
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
wnl = WordNetLemmatizer()
cleaned_texts = []
for i in range(0, len(dataset['Questions'])):
    question = dataset['Questions'][i]
    question = question.lower()
    question = question.lower()      # Change everything to lowercase
    question = re.compile('<[^<>]+>').sub(' ', question)         # Strip all HTML
    question = re.compile('[0-9]+').sub(' number ', question)        # Handle numbers
    question = re.compile('(http|https)://[^\s]').sub(' httpaddr ', question)        # Handle URLs
    question = re.compile('[^\s]+@[^\s]+').sub(' emailaddr ', question)      # Handle emails
    question = re.compile('[$]+').sub(' dollar ', question)      # Handle $ 
    question = re.compile('[^a-zA-Z0-9]').sub(' ', question)     # Handle alphanumeric
    question = re.split('[ @$/#.-:&+=[]?!(){},''">_<;%\n\r]', question)      # Remove punctuations
    question = ' '.join(question)
    tokenization = word_tokenize(question)
    question = [wnl.lemmatize(w) for w in tokenization]
    question = ' '.join(question)
    cleaned_texts.append(question)
        

# Creating the BAG OF WORDS model
from sklearn.feature_extraction.text import CountVectorizer
'''
To count total number of unique words
cv = CountVectorizer()
x = cv.fit_transform(cleaned_texts).toarray().sum(axis = 0)
'''
cv = CountVectorizer(max_features = 6200)
X = cv.fit_transform(cleaned_texts).toarray()
y = dataset.iloc[:, [1]].values

# Splitting into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling:-
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



# Part 2: Creating the ANN


# Importing Keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the first input layer and hidden layer
classifier.add(Dense(activation="relu", input_dim=6200, units=11000, kernel_initializer="uniform"))

# Adding the 2nd hidden layer
classifier.add(Dense(activation="relu", units=6000, kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Part 3: Saving the model

from sklearn.externals import joblib 

# Save the model as a pickle in a file 
joblib.dump(classifier, 'filename.pkl') 

# Load the model from the file 
classifier_from_joblib = joblib.load('filename.pkl')  

# Use the loaded model to make predictions 
classifier_from_joblib.predict(X_test)
