import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import plotly.graph_objs as go

from .templates import master_template as tp

from .collapse import *
from .functions_nn import default_plotly

lrf=10000

def layout_status():
    return html.Div(id="p2-run-button", className="page-item", children=[
        html.Button('Train RNN', id='button'),
        dcc.Interval(id='interval', interval=500),
        dcc.Interval(id='interval_counter', interval=1*1000, n_intervals=0),
        dcc.RadioItems(
            id='lock',
            options=[{'label': i, 'value': i} for i in ['Running...', 'Free']]),
        html.Div(id='output')
    ])



p2_content= \
    html.Div(className="main-container", id="main-container-p2", children=[


    html.Div(id='p2-title-button-group', children=[html.H3([html.Span("Module 2: "), "Train Recurrent Neural Network"]),
              html.Button(id='next-button-3', className="hvr-fade", children=[
                  dcc.Link('Home', href='/index', className="link-button")])
              ]),

            html.P(["The module below allows us to adjust the input settings & "
                    "layers of our recurrent neural network. This process of tuning"
                    " a neural network can be done in an automated fashion using grid search, but "
                    "for demonstration purposes, this is illustrated with an interactive dashboard. "
                    "Adjust the settings, click Train RNN, and review the results plot to see how well the model "
                    "predicts the TARGET (i.e. next event in the sequence."], className = "subtext"),
        html.Hr(),


        html.Div(className="page-group", id="p2-top-box", children=[
        html.H6(id="header-net-settings", children = ["Neural Network Settings"]),
        html.Div(className="p2-sub-box", children=[
            html.Div(id='collapse-page-item', className="page-item", children=[collapse_layout]),

            html.Div(className="slider-group", children=[

                html.Div(className='page-item small-label', children='Batch Size:'),
                html.Div(className='page-item slider-div', children=[dcc.Slider(id='p2-slider-sliderId',
                           min=10,
                           max=100,
                           marks={i: '{}'.format(i) for i in range(10, 100, 20)},
                           value=20,
                           updatemode='drag',
                           vertical=True
                           )]),
                html.Div(id='batch-size-value', className='page-item special-value', children='')
                ]),

            html.Div(className="slider-group", children=[

                html.Div(className='page-item small-label', children='Learning Rate:'),
                html.Div(className='page-item slider-div', children=[dcc.Slider(id='p2-lr',
                                                                     min=10,
                                                                     max=300,
                                                                     marks={i: '{}'.format(i/lrf) for i in
                                                                            range(100, 300, 150)},
                                                                     value=10,
                                                                     updatemode='drag',
                                                                     vertical = True
                                                                     )]),
                html.Div(id='lr-value', className='page-item special-value', children='')
            ]),

            html.Div(className="slider-group", children=[

                html.Div(className='page-item small-label', children='Epochs:'),
                html.Div(className='page-item slider-div', children=[dcc.Slider(id='p2-epoch-slider',
                                                                     min=30,
                                                                     max=100,
                                                                     marks={i: '{}'.format(i) for i in range(30, 100, 20)},
                                                                     value=40,
                                                                     updatemode='drag',
                                                                     vertical=True
                                                                     )]),
                html.Div(id="epoch-value", className='page-item special-value', children='placeholder')
        ])
        ]) ])
        ,

        html.Div( className="page-group", id="rnn_div", children=[

        html.Div(className="page-group", id="plot-and-layers", children=[
            html.Div(className="settings-box", children=[
            html.H6(id="header-net-settings", children = ["Selections"]),
            html.Div(id='submit-description', className="page-item")]),


            html.Div(className = "page-item", id='cur-plot-div', children=[dcc.Graph(id='cur_plot', figure=default_plotly(),
                                                                  style={'width':'100%'})])


             ])

                                                    ]),

        html.Div(id='button-and-console', className="page-group", children=[



                layout_status()




        ]), html.Div(id="hidden-plot", style={'display':'none'})
])

layout = tp.create_html_template(p2_content, "outer-div-p2")