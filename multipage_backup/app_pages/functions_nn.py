import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import sys



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





def generate_arrays(dataframe):
    arr_x = []
    arr_y = []
    for name, group in dataframe.groupby('user_id'):
        grp = np.array(group)
        # print("full group:\n", grp)
        last = grp[grp[:, 8] == "Last"]
        # print("last:\n",last)
        keep = grp[grp[:, 8] != "Last"]
        # print("keep:\n",keep)
        arr_x.append(keep)
        arr_y.append(last)


    data_x = np.stack(arr_x, axis=0)
    data_x = data_x[:, :, 3:7]
    data_x = data_x.astype(int)

    data_y = np.stack(arr_y, axis=0)
    data_y = data_y[:, :, 3:7]
    data_y = data_y.astype(int)

    return([data_x, data_y])

