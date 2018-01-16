# deepNeuralText
# Binary text classification based on a deep recurrent neural network
#
# Author: Michal Pikusa (pixeluz@gmail.com)
# Version 0.1

import os.path
import time
import numpy as np
import nltk
from sklearn import metrics
np.random.seed(1)
from keras.layers import Dense, Embedding
from keras.layers import LSTM, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential

def load_data(infile):
    text_file = open(infile,'r')
    text = text_file.readlines()
    text = map(str.strip,text)
    return text

def encode_data(docs):
    tokenizer = text.Tokenizer(num_words=None)
    tokenizer.fit_on_texts(docs)
    tokenized = tokenizer.texts_to_sequences(docs)
    docs = sequence.pad_sequences(tokenized, maxlen=128)
    return docs

def pos_tag(docs):
    tagged_sentences = []
    for item in docs:
        combined = []
        tokenized = nltk.word_tokenize(item)
        tagged = nltk.pos_tag(tokenized)
        for i in xrange(len(tagged)):
            combined.append('_'.join([tagged[i][0], tagged[i][1]]))
        combined_string = ' '.join(combined)
        tagged_sentences.append(combined_string)
    return tagged_sentences

def build_model(embed_size,max_length,vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=max_length))
    model.add(LSTM(50, return_sequences=True))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.1))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# Start the counter
start = time.time()

# Load raw data
print("Loading data...")
docs = load_data('corpus.txt')
labels = load_data('labels.txt')

# Tokenize and pos tag the data
print("Tokenizing and POS tagging...")
docs = pos_tag(docs)

# Encode the data
print("Encoding...")
docs_encoded = encode_data(docs)
docs_encoded = np.array(docs_encoded)
labels_encoded = np.array(labels)
labels_encoded = labels_encoded.astype(int)

# Divide the data into training(80%) and test(20%) sets
print("Dividing sets...")
msk = np.random.rand(len(docs_encoded)) < 0.8
train_data = docs_encoded[msk].astype(int)
test_data = docs_encoded[~msk].astype(int)
train_labels = labels_encoded[msk].astype(int)
test_labels = labels_encoded[~msk].astype(int)
train_labels = train_labels.reshape(len(train_labels), 1)

# Build the model
model = build_model(128,128,20000)

# If a weights file does not exist, train the network. Otherwise, load the existing weights
if os.path.isfile("weights.hdf5") == False:
    print("Training the model...")
    checkpoints = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
    model.fit(train_data, train_labels, batch_size=32, epochs=3, validation_split=0.1, callbacks=[checkpoints])
else:
    print("Loading extisting model...")
    model.load_weights("weights.hdf5")

# Test the model
print("Testing the model...")
predictions = model.predict(test_data)
rounded = [round(x[0]) for x in predictions]    # round predictions to get classes

# Print the scores
accuracy = metrics.accuracy_score(test_labels,rounded)
print("Accuracy: " + str(accuracy))

# Print out overall statistics of the run
end = time.time()
print("Finished processing in " + str(round(end - start,1)) + " seconds")