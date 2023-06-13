import json
from chatbot import tokenize,stem,bag_of_words
import numpy as np
with open ('intents.json','r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)   #We dont want to put list of arrays we make it one list and add it here
        xy.append((w,tag))

ignore_words = ['?', '!', '.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(tags)

X_train = []  # all bag of words
Y_train = []  #tag indexes
for (pattern_sentece, tag) in xy: # xy[words, tags]
    bag = bag_of_words(pattern_sentece, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    Y_train.append(label) # We have numbers for our train

X_train = np.array(X_train)
