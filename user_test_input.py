import tflearn
import numpy as np
import tensorflow as tf
import create_data_set


category = ["positive", "negtive"]
def predict_output(features):
    # Build neural network
    net = tflearn.input_data(shape=[None, len(features[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(category), activation='softmax')
    net = tflearn.regression(net)
    model = tflearn.DNN(net)
    model.load('model.tflearn')

    p = model.predict(features)
    return p

_, token = create_data_set.get_data_set()
sentence = "rahim tui ekta kharap"
features = create_data_set.get_features_of_sentence_from_user(sentence, token)
print("@@@@@///////===", features)
fetures = np.array(features)
print(predict_output([features]))
