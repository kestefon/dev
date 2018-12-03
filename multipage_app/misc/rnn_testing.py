import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import sys

import flask
from app_pages import functions_nn as fnn
from app_pages import functions_frontend as fnn
from app_pages import rnn_sequence

from sklearn.model_selection import train_test_split
import tensorflow as tf
import random
import collections
from plotnine import *

raw = pd.read_csv('https://raw.githubusercontent.com/kestefon/dev/master/data.csv')
data_list_temp = ()
data_list_temp = fnn.generate_arrays(fnn.data_cleanup(raw))

#rnn_outputs = fnn.fit_rnn(data_list_temp[0], data_list_temp[1], data_list_temp[2], data_list_temp[3])


l1={'layer':1, 'type': 'lstm', 'n_hidden':32}
l2={'layer':2, 'type': 'gru', 'n_hidden':32}
layer_list=[]
layer_list.append(l1)
layer_list.append(l2)

rnn_outputs=fnn.fit_rnn(data_inputs=data_list_temp,
                        num_epochs=100, batch_size=25, learning_rate=0.003, layers=layer_list)

#returns rnn_history, df_results)

df=rnn_outputs["history_df"]
df.head()
####
##out_plot = (ggplot(data=df) +
##                geom_point( mapping=aes(x="epoch", y="error", color="datatype") ) +
##            xlab("Epoch") + ylab("Error") + labs(color="Data"))
##

##
##l1={'layer':1, 'type': 'gru', 'n_hidden':128}
##l2={'layer':2, 'type': 'gru', 'n_hidden':128}
##layer_list=[]
##layer_list.append(l1)
##layer_list.append(l2)
##
##df
##
##for i,v in enumerate(layer_list):
##    print("index:", i, "dict-entry:", v)
##
##for i, dictionary in enumerate(layer_list):
##    for k,v in dictionary:
##        print(k,v)
##
##df=pd.DataFrame(layer_list)
##
##rnn_layers=[]
##for index, row in df.iterrows():
##    print(row['layer'], row['n_hidden'], row['type'])
##    if row['type'] == "gru":
##        cell = tf.nn.rnn_cell.GRUCell(row['n_hidden'])
##    elif row['type']=="lstm":
##        cell = tf.nn.rnn_cell.LSTMCell(row['n_hidden'])
##    else:
##        cell = tf.nn.rnn_cell.GRUCell(row['n_hidden'])
##    rnn_layers.append(cell)
##    return rnn_layers
##       
##multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
##
##
##layer_dict=collections.OrderedDict({'layer': 1, 'type': 'gru', 'n_hidden': 128})
##layer_list=[layer_dict]
