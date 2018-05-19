# deep authorship attribution
# 
# A multi-class text classification based on a deep recurrent neural network (RNN) with convolutional layers. 
# The network includes an embedding layer, so the input is first transformed into padded sequences of fixed length. 
# PoS-tagging, and additional feature engineering is also included to improve classification, but is not essential 
# for the network to run.
# 
# Version: 0.3
# 
# Author: Michal Pikusa (pikusa.michal@gmail.com)

import numpy as np
import nltk
import keras
from keras.layers import Dense, Embedding
from keras.layers import LSTM, GlobalMaxPool1D, Dropout, Conv1D, MaxPooling1D
from keras.preprocessing import text, sequence
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import pandas as pd
import spacy
from keras.utils.vis_utils import plot_model


def load_data(infile):
    df = pd.read_csv(infile)
    text = df['text'].tolist()
    text = list(map(str.strip,text))
    labels = df['author'].tolist()
    labels = list(map(str.strip,labels))
    return text,labels


def pos_tag(docs):
    nlp = spacy.load('en')
    tagged_sentences = []
    for item in docs:
        combined = []
        tagged = nlp(item)
        for token in tagged:
            combined.append('_'.join([token.lemma_, token.pos_]))
        combined_string = ' '.join(combined)
        tagged_sentences.append(combined_string)
    return tagged_sentences


def encode_data(docs):
    tokenizer = text.Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(docs)
    tokenized = tokenizer.texts_to_sequences(docs)
    docs = sequence.pad_sequences(tokenized, maxlen=128)
    return docs


def build_model(embed_size,max_length,vocab_size):
    def build_model():
        model = Sequential()
        model.add(Embedding(vocab_size, embed_size, input_length=max_length))
        model.add(Conv1D(embed_size, 7, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))
        model.add(GlobalMaxPool1D())
        model.add(Dropout(0.1))
        model.add(Dense(50, activation="relu"))
        model.add(Dropout(0.1))
        model.add(Dense(3, activation="sigmoid"))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        return model
    return build_model


print("Loading data...")
docs, labels = load_data('train.csv')

print("Tagging...")
docs = pos_tag(docs)

print("Creating features")
nn_list = []
vb_list = []
for sentence in docs:
    nns = 0
    vbs = 0
    elements = sentence.split('_')
    for item in elements:
        if item == 'NOUN':
            nns += 1
        if item == 'VERB':
            vbs += 1
    nn_list.append(nns)
    vb_list.append(vbs)
nn_list = np.array(nn_list).reshape(-1,1)
vb_list = np.array(vb_list).reshape(-1,1)

print("Encoding data...")
docs_encoded = encode_data(docs)
docs_encoded = np.array(docs_encoded)
docs_encoded = np.append(docs_encoded,nn_list,axis=1)
docs_encoded = np.append(docs_encoded,vb_list,axis=1)
labels = np.array(labels)
train_data = docs_encoded.astype(int)
encoder = LabelEncoder()
encoder.fit(labels)
encoded_labels = encoder.transform(labels)
train_labels = np_utils.to_categorical(encoded_labels)

print("Training...")
estimator = KerasClassifier(build_fn=build_model(130,130,20000),epochs=3,batch_size=128,verbose=1)
folds = KFold(n_splits=10, shuffle=True, random_state=128)
results = cross_val_score(estimator=estimator,X=train_data,y=train_labels,cv=folds)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))