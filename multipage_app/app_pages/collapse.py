
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer


collapse_layout = html.Div([

    # Button
    html.Button('Show / Hide', id='button'),
    html.Div(id='button_container', children=[


            html.Div([
                dcc.RadioItems(
                    id='choose_cell-1',
                    options=[{'label': i, 'value': i} for i in ['GRU', 'LSTM']],
                    value='GRU'
                ),
                dcc.Input(id='layer-value-1', value='', placeholder='e.g. 32, 64, 128...', type='text')]),

            html.Div([
                dcc.RadioItems(
                    id='choose_cell-2',
                    options=[{'label': i, 'value': i} for i in ['GRU', 'LSTM']],
                    value='GRU'
                ),
                dcc.Input(id='layer-value-2', value='', placeholder='Enter number...', type='text')]),


            html.Div([
                dcc.RadioItems(
                    id='choose_cell-3',
                    options=[{'label': i, 'value': i} for i in ['GRU', 'LSTM']],
                    value='GRU'
                ),
                dcc.Input(id='layer-value-3', value='', placeholder='Enter number...', type='text')]),


            html.Div([
                dcc.RadioItems(
                    id='choose_cell-4',
                    options=[{'label': i, 'value': i} for i in ['GRU', 'LSTM']],
                    value='GRU'
                ),
                dcc.Input(id='layer-value-4', value='', placeholder='Enter number...', type='text')])


        ])


    ])