import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import string
import unicodedata
import sys
import re


def read_raw_data(file_name):
    data = None
    with open(file_name) as json_data:
        data = json.load(json_data)

    return data


def create_words_tokenizer_and_tuple_set(data):
    text_with_category = []
    words_tokenizer = []
    for each_category in data.keys():
        for each_sentence in data[each_category]:
            # remove any punctuation from the sentence
            # removing the punctutation
            # each_sentence = re.sub(r'[^\w\s]', '', each_sentence)

            each_sentence = each_sentence.lower()
            print(each_sentence)
            exit()
            # extract words from each sentence and append to the word list
            w = nltk.word_tokenize(each_sentence)
            # print("tokenized words: ", w)
            words_tokenizer.extend(w)
            text_with_category.append((w, each_category))

    words_tokenizer = sorted(list(set(words_tokenizer)))
    return words_tokenizer, text_with_category


def get_features_of_sentence(token_words_of_each_sentence, words_tokenizer):
    features = [0] * len(words_tokenizer)
    for word in token_words_of_each_sentence:
        if word in words_tokenizer:
            idx = words_tokenizer.index(word)
            features[idx] += 1
    return features


def get_features_of_sentence_from_user(token_words_of_each_sentence, words_tokenizer):

    token_words_of_each_sentence = token_words_of_each_sentence.lower()
    token_words_of_each_sentence = re.sub(r'[^\w\s]', '', token_words_of_each_sentence)

    # print(each_sentence)

    # extract words from each sentence and append to the word list
    token_words_of_each_sentence = nltk.word_tokenize(token_words_of_each_sentence)
    print(token_words_of_each_sentence)
    features = [0] * len(words_tokenizer)
    for word in token_words_of_each_sentence:
        if word in words_tokenizer:
            idx = words_tokenizer.index(word)
            features[idx] += 1
    return features


def get_data_set():
    # create our training data
    training = []
    # read the data
    data = read_raw_data("bangla_data.json")
    categories = list(data.keys())
    # create an empty array for our output
    output_empty = [0] * len(categories)
    # make tokenizer and (text, cat) tuple
    words_tokenizer, text_with_category = create_words_tokenizer_and_tuple_set(data)

    for text in text_with_category:
        # list of tokenized words for the pattern
        token_words_of_each_sentence = text[0]
        # create our bag of words array
        features = get_features_of_sentence(token_words_of_each_sentence, words_tokenizer)

        output_row = list(output_empty)
        # making one hot vector
        output_row[categories.index(text[1])] = 1

        # our training set will contain a the bag of words model and the output row
        # that tells which category that bow belongs to.

        training.append([features, output_row])

    # shuffle our features and turn into np.array as tensorflow	 takes in numpy array
    random.shuffle(training)
    training = np.array(training)
    return training, words_tokenizer


get_data_set()