import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import plotly.graph_objs as go

from .templates import master_template as tp

from .collapse import *




p2_content= \
    html.Div(className="main-container", id="main-container-p2", children=[

            html.Div([ html.Span(["Module 2:"]), html.H3(["Train Recurrent Neural Network"]) ]),
            html.P(["The table below provides a preview of the data that will"
                    "feed into the model. This is anonymized user-level data"
                    "that shows us the sequence in which they clicked ads."]),

            html.Hr(className='hr'),


            html.Div(className="page-group", children=[

                html.Div(className='page-item', children='Choose test percentage:'),
                html.Div(className='page-item', children=[dcc.Slider(id='p2-slider-sliderId',
                           min=10,
                           max=40,
                           marks={i: '{}%'.format(i) for i in range(10, 40, 5)},
                           value=33,
                           updatemode='drag'
                           )]),

                html.Div(id="p2-testtrain-show", className='page-item', children='')
                ]),

        html.Div(className="page-group", children=[

            html.Div(className='page-item', children='Choose number of epochs:'),
            html.Div(className='page-item', children=[dcc.Slider(id='p2-epoch-slider',
                                                                 min=50,
                                                                 max=200,
                                                                 marks={i: '{}'.format(i) for i in range(50, 200, 50)},
                                                                 value=50,
                                                                 updatemode='drag'
                                                                 )]),

            html.Div(id="p2-epoch-show", className='page-item', children='')
        ]),

        html.Div( className="page-group", id="rnn_div", children=[html.H5('Neural Network Settings & Results'),

        html.Div([
            html.Div(className = "page-item",
                     children=[dcc.Graph(id='cur_plot')]),




            html.Div(className="page-item", children=[collapse_layout])
            ], className="page-group", id="plot-and-layers")

                                                    ]),

        html.Div(id='button-and-console', className="page-group", children=[

                html.Div(className="page-item",
                         children=[html.Button(id='p2-button-buttonId', n_clicks=0, children='Train Model')]),
                html.Div(className="page-item",
                         children=[html.Iframe(id='console-out', srcDoc='',
                                               style={'width': '75%','height':100, 'background-color': 'white'})]),

                html.Div(dcc.Interval(id='interval1', interval=1*800, n_intervals=0), style={'display': 'none'}),
                html.Div(dcc.Interval(id='interval2', interval=1*900, n_intervals=0),style={'display': 'none'}),

        ]),

        html.Div(className="page-group", children=[


                html.Div(id="p2-slider-outputDivId-hide", children='', style={'display': 'none'}),
                html.Div(id='p2-button-outputDivId-hide', style={'display': 'none'}),
                html.Div(id='p2-hidden-div-1', children='', style={'display': 'none'}),
                html.Div(id='p2-epoch-hide', children='', style={'display': 'none'}),
                html.Div(id='layer1-hidden', children='', style={'display': 'none'}),
                html.Div(id='layer2-hidden', children='', style={'display': 'none'}),
                html.Div(id='layer3-hidden', children='', style={'display': 'none'}),
                html.Div(id='layer4-hidden', children='', style={'display': 'none'})

        ])
])

layout = tp.create_html_template(p2_content, "outer-div-p2")