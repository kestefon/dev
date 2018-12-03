
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer


collapse_layout = html.Div(id='collapse', children=[


    html.Div(className='layer_container-all', children=[
        # Button
        dcc.Dropdown(id='layer-dropdown',
                     options=[
                         {'label': 'Layer 1', 'value': 'L1'},
                         {'label': 'Layer 2', 'value': 'L2'},
                         {'label': 'Layer 3', 'value': 'L3'}
                     ],
                     multi=True,
                     value=['L1'], style={'color': 'black'}
                     ),
        #layer1
            html.Div(className="page-group layer-group", id='layer-container-1', children=[
                html.Div(className="page-item cell",
                         children=[dcc.RadioItems(id='choose-cell-1',
                                                  options=[{'label': i, 'value': i.lower()} for i in ['GRU', 'LSTM']], value='gru'
                )]),
                html.Div(className="page-item",
                         children=[dcc.Input(id='layer-value-1', className='input-box', value=32, placeholder='32', type='number',
                                             min=16, max=128, step=1)]),
                html.Div(className="page-item", children=[dcc.Input(id='layer-value-1-drop', className='input-box',
                                                                    placeholder='Keep %', type='number', min=.1,
                                                                    max=1, step=.1)])

            ]),

        #layer2
            html.Div(className="page-group layer-group", id='layer-container-2', style={'display':'none'},children=[
                html.Div(className="page-item cell",
                         children=[dcc.RadioItems(id='choose-cell-2',
                                                  options=[{'label': i, 'value': i.lower()} for i in ['GRU', 'LSTM']], value='gru'
                )]),
                html.Div(className="page-item",
                         children=[dcc.Input(id='layer-value-2', className='input-box', placeholder='32', type='number',
                                             min=16, max=128, step=1)]),
                html.Div(className="page-item", children=[dcc.Input(id='layer-value-2-drop', className='input-box',
                                                                    placeholder='Keep %', type='number', min=.1,
                                                                    max=1, step=.1)])

            ]),

        #layer3
            html.Div(className="page-group layer-group", id='layer-container-3', style={'display':'none'}, children=[
                html.Div(className="page-item cell",
                         children=[dcc.RadioItems(id='choose-cell-3',
                                                  options=[{'label': i, 'value': i.lower()} for i in ['GRU', 'LSTM']], value='gru'
                )]),
                html.Div(className="page-item",
                         children=[dcc.Input(id='layer-value-3', className='input-box',  placeholder='32', type='number',
                                             min=16, max=128, step=1)]),
                html.Div(className="page-item", children=[dcc.Input(id='layer-value-3-drop', className='input-box',
                                                             placeholder='Keep %', type='number', min=.1, max=1, step=.1)])

            ])


        ])


    ])

