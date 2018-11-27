#RNN Sequence Classification Class
import functools
import sets
import tensorflow as tf
import collections
import pandas as pd

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


class VariableSequenceClassification:

    def __init__(self, data, target, learning_rate=0.003,
                 layers=[{'layer':1, 'type': 'gru', 'n_hidden':128}]):
        print("INITIALIZING")
        self.data = data
        self.target = target
        self.learning_rate = learning_rate
        self.layers=layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def length(self):
        print("activating LENGTH function")
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    # # create a BasicRNNCell
    # rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    #
    # # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
    #
    # # defining initial state
    # initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
    #
    # # 'state' is a tensor of shape [batch_size, cell_state_size]
    # outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
    #                                    initial_state=initial_state,
    #                                    dtype=tf.float32)
    #
    # # create 2 LSTMCells
    # rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]
    #
    # # create a RNN cell composed sequentially of a number of RNNCells
    # multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    #
    # # 'outputs' is a tensor of shape [batch_size, max_time, 256]
    # # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # # tf.contrib.rnn.LSTMStateTuple for each cell
    # outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
    #                                    inputs=data,
    #                                    dtype=tf.float32)

    def build_layers(self):

        df = pd.DataFrame(self.layers)

        rnn_layers = []
        list_n_hidden = []
        for index, row in df.iterrows():
            print(row['layer'], row['n_hidden'], row['type'])
            if row['type'] == "gru":
                cell = tf.nn.rnn_cell.GRUCell(row['n_hidden'])
            elif row['type'] == "lstm":
                cell = tf.nn.rnn_cell.LSTMCell(row['n_hidden'])
            else:
                cell = tf.nn.rnn_cell.GRUCell(row['n_hidden'])
            list_n_hidden.append(row['n_hidden'])
            rnn_layers.append(cell)

        final_n_hidden = list_n_hidden[-1]

        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        return [multi_rnn_cell, final_n_hidden]


    @lazy_property
    def prediction(self):
        model_layers = self.build_layers()
        print("activating PREDICTION function")
        # Recurrent network.
        output, _ = tf.nn.dynamic_rnn(
            model_layers[0],
            self.data,
            dtype=tf.float32,
            sequence_length=self.length,
        )
        last = self._last_relevant(output, self.length)
        # Softmax layer.
        weight, bias = self._weight_and_bias(
            model_layers[1], int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def cost(self):
        print("activating COST function")
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        print("activating OPTIMIZE function")
        learning_rate = self.learning_rate
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        print("activating optimizer function")
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        print("activating ERROR function")
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        print("activating WEIGHT/BIAS function")
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @staticmethod
    def _last_relevant(output, length):
        print("activating LAST RELEVANT function")
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant




