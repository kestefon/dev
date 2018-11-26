
from keras.preprocessing.text import Tokenizer
import sys
from sklearn.model_selection import train_test_split
from .rnn_sequence import *
import tensorflow as tf
import random
import numpy as np
import pandas as pd







def replaceNone(r):
    if r == None:
        return "None"
    else:
        return str(r)

def data_cleanup(df):
    data_split = df.Sequence.str.split(pat="-", expand=True)
    cleaned_data = pd.concat([df, data_split], axis=1)
    cleaned_data.drop(columns=["Sequence"], inplace=True)
    cleaned_data = pd.melt(cleaned_data, id_vars=["User ID"])
    cleaned_data.sort_values(["User ID", "variable"], inplace=True)

    # rename columns
    cleaned_data.rename(columns={'User ID': 'user_id', 'variable': 'timestep', 'value': 'event'}, inplace=True)

    #handle None values
    cleaned_data.reset_index(level=0, drop=True, inplace=True)
    cleaned_data['event'] = cleaned_data.event.apply(replaceNone)

    #tokenize
    tk = Tokenizer()
    tk.fit_on_texts(cleaned_data.event.values)
    word_index = tk.word_index
    word_index = word_index.copy()
    word_index.pop("none")

    enc = tk.texts_to_matrix(cleaned_data.event.values, mode="binary")
    enc = pd.DataFrame(enc)
    cleaned_data = pd.concat([cleaned_data, enc], axis=1)
    cleaned_data['next_event'] = cleaned_data['event'].shift(-1)

    def checkDict(event, next_event, timestep):
        if event.lower() in word_index and next_event.lower() not in word_index:
            return "Last Event"
        elif timestep == 6 and event.lower() in word_index:
            return "Last Event"
        else:
            return "-"

    cleaned_data.next_event.fillna("None", inplace=True)
    cleaned_data['flag'] = cleaned_data.apply(lambda x: checkDict(x['event'], x['next_event'], x['timestep']), axis=1)
    cleaned_data.drop(columns=[0, 1], inplace=True)

    return cleaned_data





def generate_arrays(dataframe, var_test_percent=0.33, var_rand_state=42):
    arr_x = []
    arr_y = []
    for name, group in dataframe.groupby('user_id'):
        grp = np.array(group)
        # print("full group:\n", grp)
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

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=var_test_percent, random_state=var_rand_state)

    return(train_x, train_y, test_x, test_y)






def fit_rnn(X_train,y_train,X_test,y_test):
    # We treat images as sequences of pixel rows.
    _, rows, row_size = X_train.shape
    print(X_train.shape)
    num_classes = X_train.shape[2]
    data = tf.placeholder(tf.float32, [None, rows, row_size])
    target = tf.placeholder(tf.float32, [None, num_classes])
    model = VariableSequenceClassification(data, target)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[2]))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[2]))
    rnn_history = ''
    list_results=[]

    for epoch in range(50):

        for _ in range(20):
            rand_idx = random.sample(range(0, X_train.shape[0]), 10)
            print(_,len(rand_idx))
            batch_data = X_train[rand_idx]  # sample from data
            # print("Batch data:\n", batch_data)
            batch_target = y_train[rand_idx]
            # print("Batch target:\n", batch_target)
            print("starting SESS.RUN")
            print("Batch shapes:", batch_data.shape, batch_target.shape)
            sess.run(model.optimize, {data: batch_data, target: batch_target})
        error = sess.run(model.error, {data: X_test, target: y_test})
        dict_error={'epoch': epoch, 'error': error}
        list_results.append(dict_error)

        line='Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error)
        rnn_history=rnn_history + line + '<BR>'
        print(line)
    df_results = pd.DataFrame(list_results)
    return(rnn_history, df_results, model)

