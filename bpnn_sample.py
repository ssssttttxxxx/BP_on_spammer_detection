import tensorflow as tf
import numpy as np
import networkx as nx
import copy
from sklearn.model_selection import train_test_split

training_epochs = 500
n_neurons_in_h1 = 60
n_neurons_in_h2 = 60
learning_rate = 0.01

n_features = 5
n_classes = 2

attributes_name = ['reviewerID', 'friends_num', 'reviews_num', 'photo_num', ]
shuffle_stat = 42
trainset_size = 0.8


def split_train_test(graph, attributes):

    X_list = list()
    Y_list = list()
    for i, node in enumerate(graph.nodes()):
        temp_list = list()
        for attr_name, val in graph.node[node].items():

            if attr_name in attributes:
                temp_list.append(val)
            elif attr_name == 'fake':
                Y_list.append(val)

        if 'degree' in attributes:
            degree = graph.degree(node)
            temp_list.append(int(degree))

        X_list.append(temp_list)
    X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=1 - trainset_size,
                                                        random_state=shuffle_stat)

    return X_train, X_test, Y_train, Y_test


X = tf.placeholder(tf.float32, [None, n_features], name='features')
Y = tf.placeholder(tf.float32, [None, n_classes], name='labels')

W1 = tf.Variable(tf.truncated_normal([n_features, n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)),
                 name='weights1')
b1 = tf.Variable(tf.truncated_normal([n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)), name='biases1')

y1 = tf.nn.tanh((tf.matmul(X, W1) + b1), name='activationLayer1')

# network parameters(weights and biases) are set and initialized(Layer2)
W2 = tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2], mean=0, stddev=1 / np.sqrt(n_features)),
                 name='weights2')
b2 = tf.Variable(tf.random_normal([n_neurons_in_h2], mean=0, stddev=1 / np.sqrt(n_features)), name='biases2')
# activation function(sigmoid)
y2 = tf.nn.sigmoid((tf.matmul(y1, W2) + b2), name='activationLayer2')

# output layer weights and biasies
Wo = tf.Variable(tf.random_normal([n_neurons_in_h2, n_classes], mean=0, stddev=1 / np.sqrt(n_features)),
                 name='weightsOut')
bo = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=1 / np.sqrt(n_features)), name='biasesOut')
# activation function(softmax)
a = tf.nn.softmax((tf.matmul(y2, Wo) + bo), name='activationOutputLayer')

# cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(a), reduction_indices=[1]))
# optimizer
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# compare predicted value from network with the expected value/target
correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(Y, 1))
# accuracy determination
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

initial = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initial)

    path = 'graph/friendship_attr_total_cleanUnknown_cleanDegree0.pickle'
    graph = nx.read_gpickle(path)
    X_train, X_test, Y_train, Y_test = split_train_test(graph, attributes_name)

    tr_feature = [X[1:] for X in X_train]
    tr_label = Y_train

    ts_feature = [X[1:] for X in X_test]
    ts_label = Y_test




