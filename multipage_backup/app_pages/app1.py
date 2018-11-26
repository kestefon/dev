import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer

layout = html.Div([
    html.H3(children="Page 1: Data Preview"),
    dcc.Link('Go to App 1', href='/page-1'),
    dcc.Link('Go to App 2', href='/page-2'),
    dcc.Link('Go to App 3', href='/page-3'),
    dcc.Dropdown(id='p1-id-dropdown-data',
        options=[
            {'label': 'Raw Data', 'value': 'RAW'},
            {'label': u'Cleaned Data', 'value': 'CLEAN'}
        ],
        value='RAW'
    ),
    html.Button(id='p1-submit-button', n_clicks=0, children='Submit'),
    html.Div(id='p1-intermediate-value', style={'display': 'none'}),
    html.Div(id='p1-output-state', style={'display': 'none'}),
    html.H4(children='Data Preview'),
    html.Table(id='p1-table'),
    html.H1(id='p1-div-out', children='')

])

