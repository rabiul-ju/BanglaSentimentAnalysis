import tflearn
import tensorflow as tf
import create_data_set

train_data = create_data_set.get_data_set()
test_data = train_data[:2]
train_data = train_data[2:]

train_x = list(train_data[:, 0])
train_y = list(train_data[:, 1])

test_x = list(test_data[:, 0])
test_y = list(test_data[:, 1])

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, validation_set=(test_x, test_y), batch_size=8, show_metric=True)
model.save('model.tflearn')

