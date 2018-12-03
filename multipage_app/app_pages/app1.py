import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.graph_objs as go

from .functions_nn import *


from .templates import master_template as tp

from rq import Queue
from worker import conn

q = Queue('high', is_async=False, connection=conn, default_timeout=60*3)





p1_content = html.Div(id="main-container-p1", className="main-container", children=[
                                html.Div([html.Button(id='next-button-2', className="hvr-fade", children=[
                                        dcc.Link('Next Page', href='/page-2', className="link-button")
                                    ]) ]),
                                    html.H3(children="Module 1: Sequence Data Preview"),
                                    html.P([
html.H6("Data Structure:", className='subheader-2'),
html.P("Depending on the specific approach, sequence data is typically structured "
       "in one of two ways:", className='subtext'),
html.Ul([
html.Li("Text format, where the sequence data is processed and analyzed as text data. "
        "Each event becomes a token (analogous to a word or a character), "
        "and is identified by a delimiter (i.e. a dash, in the example above)."),
html.Li("Sparse matrix/tensor format. In this format, the data is converted "
        "to multi-dimensional arrays of 1s and 0s. For sequence analysis,"
        " this is typically a 3D tensor, with axes (dimensions) representing users, timestep, "
        "and features of the event.")
], className='subtext')

                                    ]),
                                    html.Hr(),
                                    dcc.Dropdown(id='p1-top-dropdown',
                                        options=[
                                            {'label': 'Data Format A: Text Representation', 'value': 'RAW'},
                                            {'label': u'Data Format B: One-Hot Encoding', 'value': 'CLEAN'},
                                            {'label': u'Final Format for RNN: 3D Tensors', 'value': 'TENSOR'}
                                        ],
                                        value='RAW',
                                                 style = {
                                                     'color': 'black'
                                                 }
                                    ),


                                    html.Div(id='p1-intermediate-value', style={'display': 'none'}),
                                    html.Div(id='p1-output-state', style={'display': 'none'}),

                                html.Div(id="p1-toggle-table", children=[html.Table(id='p1-table')]),
                                html.Div(id="p1-toggle-graph", children=[dcc.Graph(id='3d-graph',
                                                                                   figure=q.enqueue(generate_3d).result)])
                                ])




layout = tp.create_html_template(p1_content, "outer-div-p1")


# one output div, if then logic to serve up a different layout based on toggle.