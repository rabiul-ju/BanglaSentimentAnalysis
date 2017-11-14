# -*- coding: utf-8 -*-

import tflearn
import numpy as np
import tensorflow as tf
import create_data_set_bangla


category = ["Happy", "Sorrow", "Surprised", "Depressed" ]


def predict_output(features):
    # Build neural network
    net = tflearn.input_data(shape=[None, len(features[0])])
    net = tflearn.fully_connected(net, 100)
    net = tflearn.fully_connected(net, 100)
    net = tflearn.fully_connected(net, len(category), activation='softmax')
    net = tflearn.regression(net)
    model = tflearn.DNN(net)
    model.load('bangla_model1.tflearn')

    p = model.predict(features)
    return p


_, token = create_data_set_bangla.get_data_set()
"""
>>>import codecs
>>>f = codecs.open("test", "r", "utf-8")
codecs.open(file,'w','utf-8')
# """
# import codecs
# token = codecs.open("outfile")
sentence = u'তৌসিফ ভাই এর নাটক খারাপ আর হবে কিভাবে'
features = create_data_set_bangla.get_features_of_sentence_from_user(sentence, token)
print("@@@@@///////===", features)
print(category)
features = np.array(features)
print(predict_output([features]))
