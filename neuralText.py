# neuralText
# A simple binary text classification based on a feed-forward neural network
#
# Author: Michal Pikusa (pixeluz@gmail.com)
# Version 0.1

import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

def load_data(infile):
    text_file = open(infile,'r')
    text = text_file.readlines()
    text = map(str.strip,text)
    return text

def preprocess(dataset):
    lemmatized_list = []
    lemmatizer = WordNetLemmatizer()
    stop = set(stopwords.words('english'))
    for line in dataset:
        line = ''.join(line)
        no_stops = [i for i in line.lower().split() if i not in stop]
        lemmatized = map(lambda x: lemmatizer.lemmatize(x), no_stops)
        lemmatized_list.append(' '.join(l for l in lemmatized if l not in string.punctuation))
    return lemmatized_list

def encode_data(dataset):
    vec = CountVectorizer()
    data = vec.fit_transform(dataset).toarray()
    return data

def sigmoid(x, d=False):
    if (d == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def train_network(X, y, dim_1, epochs, alpha):
    syn0 = 2 * np.random.random((dim_1, 1)) - 1
    for iter in xrange(epochs):
        l0 = X
        l1 = sigmoid(np.dot(l0, syn0))
        l1_error = y - l1
        print "Epoch: " + str(iter) + "/" + str(epochs) + " Error: " + str(np.mean(np.abs(l1_error)))
        l1_delta = l1_error * sigmoid(l1, True)
        syn0 += alpha * np.dot(l0.T, l1_delta)
    return syn0

def test_network(inputs, syn0):
    l1 = sigmoid(np.dot(inputs, syn0))
    return l1

# Start the counter and set the seed for reproducibility
start = time.time()
np.random.seed(1)

# Load raw data
print "Loading data..."
docs = load_data('corpus.txt')
labels = load_data('labels.txt')

# Preprocess the data
print "Preprocessing..."
docs_preprocessed = preprocess(docs)

# Encode the data
print "Encoding..."
docs_encoded = encode_data(docs_preprocessed)
labels_encoded = np.array(labels)
labels_encoded = labels_encoded.astype(int)

# Divide the data into training(80%) and test set(20%)
print "Dividing sets..."
msk = np.random.rand(len(docs_encoded)) < 0.8
train = docs_encoded[msk].astype(int)
test = docs_encoded[~msk].astype(int)
train_labels = labels_encoded[msk].astype(int)
test_labels = labels_encoded[~msk].astype(int)

# Train the network
print "Training the model..."
dim = np.shape(train)[1]
train_labels = train_labels.reshape(len(train_labels), 1)
model = train_network(train,train_labels,dim,100,1)

# Test the model
print "Testing the model..."
test_results = test_network(test,model)
test_results = np.round(test_results,decimals=0)
predictions = test_results.astype(int)

# Print the scores
accuracy = metrics.accuracy_score(test_labels,predictions)
print "Accuracy: " + str(accuracy)

# Save the model
print "Saving the model..."
pickle.dump(model, open("model.pickle", "wb"))

# Print out overall statistics of the run
end = time.time()
print "Finished processing in " + str(round(end - start,1)) + " seconds"