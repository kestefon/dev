import random
import functools
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.text import Tokenizer


# RNN Sequence Classification Class
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
                 layers=[{'layer': 1, 'type': 'gru', 'n_hidden': 32}]):
        print("INITIALIZING")
        self.data = data
        self.target = target
        self.learning_rate = learning_rate
        self.layers = layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def length(self):
        print("activating LENGTH function")
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        print('used:', used)
        length = tf.reduce_sum(used, reduction_indices=1)
        print("Length1", length)
        length = tf.cast(length, tf.int32)
        print("Length2", length)
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
        print("run build_layers in rnn_sequence.py")
        df = pd.DataFrame(self.layers)
        print("df:", df)
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
        print("final_n_hidden:", final_n_hidden)
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        print("multi_rnn_cell:", multi_rnn_cell)
        return [multi_rnn_cell, final_n_hidden]

    @lazy_property
    def prediction(self):
        model_layers = self.build_layers()
        print("activating PREDICTION function")
        print("model_layers 0", model_layers[0])
        # Recurrent network.
        output, _ = tf.nn.dynamic_rnn(
            model_layers[0],
            self.data,
            dtype=tf.float32,
            sequence_length=self.length,
        )
        print("output of prediction:", output)
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




###DATA HANDLER CLASS



class DataHandler:
    def __init__(self, df, test_size=0.33, rand_state=42):
        self.hello_attribute = "hello"
        self.df = df
        self.test_size = test_size
        self.rand_state = rand_state

    def data_cleanup(self):
        df = self.df
        data_split = df.Sequence.str.split(pat="-", expand=True)
        df = pd.concat([df, data_split], axis=1)
        df.drop(columns=["Sequence"], inplace=True)
        df = pd.melt(df, id_vars=["User ID"])
        df.sort_values(["User ID", "variable"], inplace=True)

        # rename columns
        df.rename(columns={'User ID': 'user_id', 'variable': 'timestep', 'value': 'event'}, inplace=True)

        def replaceNone(replace_val):
            if replace_val == None:
                return "None"
            else:
                return str(replace_val)

        # handle None values
        df.reset_index(level=0, drop=True, inplace=True)
        df['event'] = df.event.apply(replaceNone)

        # tokenize
        tk = Tokenizer()
        tk.fit_on_texts(df.event.values)
        word_index = tk.word_index
        word_index = word_index.copy()
        word_index.pop("none")

        enc = tk.texts_to_matrix(df.event.values, mode="binary")
        enc = pd.DataFrame(enc)
        df = pd.concat([df, enc], axis=1)
        df['next_event'] = df['event'].shift(-1)

        def checkDict(df, event, next_event, timestep):
            if event.lower() in word_index and next_event.lower() not in word_index:
                return "Last Event"
            elif timestep == max(df.timestep) and event.lower() in word_index:
                print(max(df.timestep))
                return "Last Event"
            else:
                return "-"

        df.next_event.fillna("None", inplace=True)
        df['flag'] = df.apply(lambda x: checkDict(df, x['event'], x['next_event'], x['timestep']), axis=1)
        df.drop(columns=[0, 1], inplace=True)
        # This is the cleaned df that will be displayed as a table

        ##        print("Dataframe", df)
        ##        print("Word Index", word_index)

        arr_x = []
        arr_y = []
        for name, group in df.groupby('user_id'):
            grp = np.array(group)
            # print("name:", name, "full group:\n", grp)
            last = grp[grp[:, 8] == "Last Event"]
            # print("last:\n",last)
            keep = grp[grp[:, 8] != "Last Event"]
            # print("keep:\n",keep)
            arr_x.append(keep)
            arr_y.append(last)

        data_x = np.stack(arr_x, axis=0)
        data_x = data_x[:, :, 3:7]
        data_x = data_x.astype(int)

        data_y = np.stack(arr_y, axis=0)
        data_y = data_y[:, :, 3:7]
        data_y = data_y.astype(int)

        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=self.test_size,
                                                            random_state=self.rand_state)
        # return the formatted dataset and the test/train arrays
        print("train_x shape: ", train_x.shape)
        print("train_y shape: ", train_y.shape)
        print("test_x shape: ", test_x.shape)
        print("test_y shape: ", test_y.shape)

        _, rows, row_size = train_x.shape
        print("train_x shape: ", train_x.shape)
        num_classes = train_x.shape[2]

        print("train_x.shape[2], i.e. num_classes: ", num_classes)

        data = tf.placeholder(tf.float32, [None, rows, row_size])
        target = tf.placeholder(tf.float32, [None, num_classes])

        train_y = train_y.reshape((train_y.shape[0], train_y.shape[2]))
        test_y = test_y.reshape((test_y.shape[0], test_y.shape[2]))
        print("train_y shape: ", train_y.shape)
        print("test_y shape: ", test_y.shape)

        print(
            "Returning a list with 1df and 2 lists. out[1] is an pre-processed dataframe with a word_index. out[2] has the test/train arrays")
        return {'raw': self.df, 'clean': [df, word_index], 'tf_data': data, 'tf_target': target,
                'tensors': [train_x, train_y, test_x, test_y], 'tensor_info': [_, rows, row_size]}

###############
model trainning function
def train_model(self, data, target, tensors, row_info, num_epochs=50, learning_rate=0.003, batch_size=25):

    list_results = []
    out_string = '''
            Generated {} tensors.
            Dimensions of training data: {}
            Dimensions of training target: {}
            Dimensions of test data: {}
            Dimensions of test target: {}

            Printing results every 10 epochs:
            '''.format(len(tensors), tensors[0].shape,
                       tensors[1].shape,
                       tensors[2].shape, tensors[3].shape)

    # INITIALIZE MODEL
    model = VariableSequenceClassification(data=data, target=target,
                                           learning_rate=learning_rate)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    _, rows, row_size = row_info
    train_x, train_y, test_x, test_y = tensors

    # TRAIN MODEL
    rnn_history = out_string
    for epoch in range(num_epochs):

        for _ in range(int(train_x.shape[0] / batch_size) + 1):
            rand_idx = random.sample(range(0, train_x.shape[0]), batch_size)
            # print(_,len(rand_idx))

            batch_data = train_x[rand_idx]  # sample from data
            # print("Batch data:\n", batch_data)

            batch_target = train_y[rand_idx]
            # print("Batch target:\n", batch_target)
            # print("starting SESS.RUN")
            # print("Batch shapes:", batch_data.shape, batch_target.shape)
            sess.run(model.optimize, {data: batch_data, target: batch_target})

        error = sess.run(model.error, {data: test_x, target: test_y})
        dict_error = {'datatype': 'test', 'epoch': epoch, 'error': error}
        list_results.append(dict_error)

        error_train = sess.run(model.error, {data: train_x, target: train_y})
        dict_error_train = {'datatype': 'train', 'epoch': epoch, 'error': error_train}
        list_results.append(dict_error_train)

        line = 'Test Error: Epoch {:2d} yielded error {:3.1f}%\nTraining Error: ' \
               'Epoch {:2d} yielded error {:3.1f}%'.format(epoch + 1, 100 * error, epoch + 1, 100 * error_train)

        rnn_history = rnn_history + line + '\n'

        if (epoch + 1) % 10 == 0:
            print(line, sep="")

    df_results = pd.DataFrame(list_results)
    print("df_results:", df_results.head())

    return {'history_text': rnn_history, 'history_df': df_results}

