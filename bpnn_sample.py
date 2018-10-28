import tensorflow as tf
import numpy as np
import networkx as nx
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
training_epochs = 50
n_neurons_in_h1 = 60
n_neurons_in_h2 = 60
learning_rate = 0.01

n_features = 3
n_classes = 2

attributes_name = ['reviewerID', 'friends_num', 'reviews_num', 'photo_num',]
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
    X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=1 - trainset_size)

    return X_train, X_test, Y_train, Y_test


X = tf.placeholder(tf.float32, [None, n_features], name='features')
Y = tf.placeholder(tf.float32, [None, n_classes], name='labels')
keep_prob = tf.placeholder("float")

W1 = tf.Variable(tf.truncated_normal([n_features, n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)),
                 name='weights1')
b1 = tf.Variable(tf.truncated_normal([n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)), name='biases1')
y1 = tf.nn.tanh((tf.matmul(X, W1) + b1), name='activationLayer1')

W2 = tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2], mean=0, stddev=1 / np.sqrt(n_features)),
                 name='weights2')
b2 = tf.Variable(tf.random_normal([n_neurons_in_h2], mean=0, stddev=1 / np.sqrt(n_features)), name='biases2')
y2 = tf.nn.sigmoid((tf.matmul(y1, W2) + b2), name='activationLayer2')

Wo = tf.Variable(tf.random_normal([n_neurons_in_h2, n_classes], mean=0, stddev=1 / np.sqrt(n_features)),
                 name='weightsOut')
bo = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=1 / np.sqrt(n_features)), name='biasesOut')
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

    path = 'graph/friendship_reviewer_label_attr_clean_unknown.pickle'
    graph = nx.read_gpickle(path)
    X_train, X_test, Y_train, Y_test = split_train_test(graph, attributes_name)

    tr_feature = [X[1:] for X in X_train]
    tr_label_to_onehot = tf.one_hot(Y_train, n_classes, 1, 0)
    tr_label = sess.run(tr_label_to_onehot)
    ts_feature = [X[1:] for X in X_test]
    ts_label_to_onehot = tf.one_hot(Y_test, n_classes, 1, 0)
    ts_label = sess.run(ts_label_to_onehot)
    batchsize = 100

    for epoch in range(training_epochs):
        for i in range(0, len(tr_feature), batchsize):
            # print('epoch:', epoch, 'batch: ', i)

            start = i
            end = start + batchsize

            x_batch = tr_feature[start:end]
            y_batch = tr_label[start:end]

            sess.run(train_step, feed_dict={X: x_batch, Y: y_batch, keep_prob: 0.2})

        y_pred = sess.run(tf.argmax(a, 1), feed_dict={X: ts_feature, keep_prob: 1.0})
        y_true = sess.run(tf.argmax(ts_label, 1))
        acc = sess.run(accuracy, feed_dict={X: ts_feature, Y: ts_label, keep_prob: 1.0})

        # for i, j in zip(y_pred, y_true):
        #     if i == 0 and j == 1:
        #         print('there is mistake 1')

        print('recall rate', recall_score(y_true, y_pred))
        print('epoch', epoch, acc)
        print('---------------')
        # print(y_pred, y_true)
