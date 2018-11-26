import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer

layout = html.Div([
    html.H3(children="App 2"),
    dcc.Link('Go to App 1', href='/page-1'),
    dcc.Link('Go to App 2', href='/page-2'),
    dcc.Link('Go to App 3', href='/page-3'),
    html.Div([
        html.Button(id='p2-id-button-arrays', n_clicks=0, children='Generate Arrays')]
    ),
    html.H4(children='Choose Train/Test % Split'),
    dcc.Slider(id='p2-id-slider-split',
        min=15,
        max=50,
        marks={i: '{}%'.format(i) for i in range(15,51,5)},
        value=30,
        updatemode='drag'
    ),
    html.Div(id="p2-id-div-split", children=''),
    html.Button(id='p2-id-button-split', n_clicks=0, children='Split Dataset'),
    html.Div(id='p2-intermediate-value', style={'display': 'none'}),
    html.Div(id='p2-output-state', style={'display': 'none'}),
    html.Div(id='p2-output-state-arrays', style={'display': 'none'}),
    dcc.Interval(id='p2-interval2', interval=60 * 1000, n_intervals=0),
    html.H1(id='p2-div-out', children=''),
    html.Iframe(id='p2-console-out',srcDoc='',style={'width':
'100%','height':200})
])





