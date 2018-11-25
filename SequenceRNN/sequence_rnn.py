import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
import functools
import sets
import tensorflow as tf
import random
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from io import BytesIO
import base64
import pandas as pd
import numpy as np

raw=pd.read_csv('https://raw.githubusercontent.com/kestefon/dev/master/data.csv')

#file='data.csv'

# data I/O
#raw = open(file, 'r').read() # should be simple plain text file
#raw =pd.read_csv(file)
##seq=data.iloc[:,[1]]
##seq=seq.Sequence.str.split(pat="-",expand=True)
####chars = list(set(data))
####data_size, vocab_size = len(data), len(chars)
####print('data has %d characters, %d unique.' % (data_size, vocab_size))
####char_to_ix = { ch:i for i,ch in enumerate(chars) }
####ix_to_char = { i:ch for i,ch in enumerate(chars) }
####
##vals=pd.unique(seq.Sequence.values)
##
##import re
##vals.replace(r"<\n\'\]\[>","")
##
##vals=str(vals)
##vals=re.sub(r"[\n\'\]\[]", "", vals)
##vals=re.sub(r"\-", " ", vals)
##vals=vals.split(" ")
##vals=set(vals)
##vals_dict={ch:i for i,ch in enumerate(vals)}
##vocab_len=len(vals)





data_split=raw.Sequence.str.split(pat="-",expand=True)
data_final=pd.concat([raw,data_split],axis=1)
data_final.drop(columns=["Sequence"],inplace=True)
data_final=pd.melt(data_final, id_vars=["User ID"])
data_final.sort_values(["User ID", "variable"], inplace=True)

#rename columns
data_final.rename(columns={'User ID': 'user_id', 'variable':'timestep' , 'value':'event'}, inplace=True)

#filter NoneTypes
#data_filtered=data_final[(data_final.event.values!=None)]


def replaceNone(r):
    if r==None:
        return "None"
    else:
        return str(r)



data_final.reset_index(level=0, drop=True, inplace=True)
data_final['event']=data_final.event.apply(replaceNone)

tk=Tokenizer()
tk.fit_on_texts(data_final.event.values)
word_index=tk.word_index
word_index=word_index.copy()
word_index.pop("none")


enc=tk.texts_to_matrix(data_final.event.values,mode="binary")
enc=pd.DataFrame(enc)
new_data=pd.concat([data_final,enc],axis=1)
new_data['next_event'] = new_data['event'].shift(-1)

def checkDict(event, next_event, timestep):
    if event.lower() in word_index and next_event.lower() not in word_index:
        return "Last"
    elif timestep==6 and event.lower() in word_index:
        return "Last"
    else:
        return "Keep"



    
new_data.next_event.fillna("None",inplace=True)
new_data['flag']= new_data.apply(lambda x: checkDict(x['event'],x['next_event'],x['timestep']),axis=1)
new_data.drop(columns=[0,1],inplace=True)

arr_x=[]
arr_y=[]
for name, group in new_data.groupby('user_id'):
    grp=np.array(group)
    #print("full group:\n", grp)
    last=grp[grp[:,8]=="Last"]
    #print("last:\n",last)
    keep=grp[grp[:,8]!="Last"]
    #print("keep:\n",keep)
    arr_x.append(keep)
    arr_y.append(last)

##shape_ls=[]
##for i,arr in enumerate(arr_x):
##    shape_ls.append(arr.shape)
##
##t3d[t3d[:,:,9] == "Last"]

data_x=np.stack(arr_x,axis=0)
data_x=data_x[:,:,3:7]
data_x=data_x.astype(int)

data_y=np.stack(arr_y,axis=0)
data_y=data_y[:,:,3:7]
data_y=data_y.astype(int)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_x,data_y, test_size=0.33, random_state=42)



# Working example for my blog post at:
# http://danijar.com/variable-sequence-lengths-in-tensorflow/
import functools
import sets
import tensorflow as tf
#from tensorflow.models.rnn import rnn_cell
#from tensorflow.models.rnn import rnn


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

    def __init__(self, data, target, num_hidden=128, num_layers=3):
        print("INITIALIZING")
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
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

    @lazy_property
    def prediction(self):
        print("activating PREDICTION function")
        # Recurrent network.
        output, _ = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.GRUCell(self._num_hidden),
            data,
            dtype=tf.float32,
            sequence_length=self.length,
        )
        last = self._last_relevant(output, self.length)
        # Softmax layer.
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.target.get_shape()[1]))
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
        learning_rate = 0.001
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

 

if __name__ == '__main__':
    # We treat images as sequences of pixel rows.
    _, rows, row_size = X_train.shape
    num_classes = X_train.shape[2]
    data = tf.placeholder(tf.float32, [None, rows, row_size])
    target = tf.placeholder(tf.float32, [None, num_classes])
    model = VariableSequenceClassification(data, target)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    y_train=y_train.reshape((y_train.shape[0],y_train.shape[2]))
    y_test=y_test.reshape((y_test.shape[0],y_test.shape[2]))
    for epoch in range(100):      

        for _ in range(50):
            rand_idx=random.sample(range(0,X_train.shape[0]),10)
            #print(rand_idx)
            batch_data = X_train[rand_idx] #sample from data
            #print("Batch data:\n", batch_data)
            batch_target=y_train[rand_idx]
            #print("Batch target:\n", batch_target)
            
            #print("starting SESS.RUN")
            #print(batch_data.shape, batch_target.shape)
            sess.run(model.optimize, {data: batch_data, target: batch_target})
        error = sess.run(model.error, {data: X_test, target: y_test})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
##


