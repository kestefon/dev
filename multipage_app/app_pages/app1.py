import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer



layout = html.Div(className = "layout_container", children=[html.Div(
    className='menu-container',
                  children=[
                      html.Div(className="menu", children=
                                     [html.Div(className="date",children=["Aug 14, 2016"]),
                                      html.Div(className="links",children=[
                                          html.Div(className="signup",children=["Sign Up"]),
                                          html.Div(className="login",children=["Login"])
                                      ])

                                    ])
                  ]),

            html.Div(className="main-page", children=[
                html.Div(className="main-container", children=[
                                            html.Div([
                                                html.H3(children="Page 1: Data Preview"),
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
                                                               ]),

                html.Div(className="box-grid-container", children=[
                        html.Div(className="box box1", children=[dcc.Link('Go to Module 1', href='/page-1')]),
                        html.Div(className="box box2", children=[dcc.Link('Go to Module 2', href='/page-2')]),
                        html.Div(className="box box3", children=[dcc.Link('Go to Module 3', href='/page-3')]),
                        html.Div(className="box box3", children=[dcc.Link('Go to HOME', href='/index')]),
                ])
            ])

])

