import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
import functools
import sets
import tensorflow as tf
##from tensorflow.models.rnn import rnn_cell
##from tensorflow.models.rnn import rnn
file='data.csv'

# data I/O
data = open(file, 'r').read() # should be simple plain text file
data=pd.read_csv(file)
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





data_split=data.Sequence.str.split(pat="-",expand=True)
data_final=pd.concat([data,data_split],axis=1)
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


arr_x=[]
arr_y=[]
for name, group in new_data.groupby('user_id'):
    grp=np.array(group)
    #print("full group:\n", grp)
    last=grp[grp[:,9]=="Last"]
    #print("last:\n",last)
    keep=grp[grp[:,9]!="Last"]
    #print("keep:\n",keep)
    arr_x.append(keep)
    arr_y.append(last)

shape_ls=[]
for i,arr in enumerate(arr_x):
    shape_ls.append(arr.shape)

t3d[t3d[:,:,9] == "Last"]

data_x=np.stack(arr_x,axis=0)
data_y=np.stack(arr_y,axis=0)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.33, random_state=42)
test=t3d[0]
#test=np.delete(test,(1),axis=0)


##def length(sequence):
##  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
##  length = tf.reduce_sum(used, 1)
##  length = tf.cast(length, tf.int32)
##  return length
##
##def last_relevant(output, length):
##  batch_size = tf.shape(output)[0]
##  max_length = tf.shape(output)[1]
##  out_size = int(output.get_shape()[2])
##  index = tf.range(0, batch_size) * max_length + (length - 1)
##  flat = tf.reshape(output, [-1, out_size])
##  relevant = tf.gather(flat, index)
##  return relevant
##
##def get_dataset():
##    """Read dataset and flatten images."""
##    dataset = sets.Ocr()
##    dataset = sets.OneHot(dataset.target, depth=2)(dataset, columns=['target'])
##    dataset['data'] = dataset.data.reshape(
##        dataset.data.shape[:-2] + (-1,)).astype(float)
##    train, test = sets.Split(0.66)(dataset)
##    return train, test

##    
##new_arr=np.zeros((len(new_data.user_id.unique()),max(new_data.timestep)+1,8))



##data_filtered=data_filtered.to_records()
##new_dt = np.dtype(data_filtered.dtype.descr + [('enc0','i8'), ('enc1','i8'), ('enc2','i8'), ('enc3','i8'), ('enc4','i8')])
##
##
##
##tk=Tokenizer()
##tk.fit_on_texts(data_filtered['event'])
##data_matrix=tk.texts_to_matrix(data_filtered['event'],mode="binary")
##word_index=tk.word_index
##
##new_data=np.zeros(data_filtered.shape, dtype=new_dt)
##new_data['index'] = data_filtered['index']
##new_data['user_id'] = data_filtered['user_id']
##new_data['timestep'] = data_filtered['timestep']
##new_data['event'] = data_filtered['event']
##
##new_data['enc0'] = data_matrix[:,0]
##new_data['enc1'] = data_matrix[:,1]
##new_data['enc2'] = data_matrix[:,2]
##new_data['enc3'] = data_matrix[:,3]
##new_data['enc4'] = data_matrix[:,4]
###reshape into tensor
##n_users=len(data_filtered.user_id.unique())
##n_timestep=max(data_filtered.timestep)
##



### reshape 2D array
##from numpy import array
### list of data
##data = [[11, 22],
##		[33, 44],
##		[55, 66]]
### array of data
##data = array(data)
##print(data.shape)
### reshape
##data = data.reshape((data.shape[0], data.shape[1], 1))
##print(data.shape)
##
##data_filtered.reset_index(level=0, inplace=True)

##cols_exclude="Sequence"
##[i for i in list(test_concat.columns) if i not in [cols_exclude]]

##samples=["The cat sat on the mat.","The dog ate my homework."]
##token_index={}
##for sample in samples:
##    for word in sample.split():
##        if word not in token_index:
##            token_index[word] = len(token_index) + 1
##
##
##results = np.zeros(shape=(len(samples),
##                          10,
##                          max(token_index.values())+1))
##
##for i, sample in enumerate(samples):
##    for j, word in list(enumerate(sample.split()))[:10]:
##        index=token_index.get(word)
##        results[i,j,index]=1
##
##
##for i, sample in enumerate(samples):
##    for j, word in list(enumerate(sample.split()))[:10]:
##        index=token_index.get(word)
##        print(i,sample,j,word)
##        print("word:",word,"\nindex:",index)
##        print("will update results[",i,j,index,"to be 1")
