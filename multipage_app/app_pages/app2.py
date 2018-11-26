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
                                                            html.H3(children="Module 2"),
                                                            html.H4(children='Choose Train/Test % Split'),

                                                            dcc.Slider(id='p2-slider-sliderId',
                                                                       min=15,
                                                                       max=50,
                                                                       marks={i: '{}%'.format(i) for i in range(15, 51, 5)},
                                                                       value=30,
                                                                       updatemode='drag'
                                                                       ),

                                                            html.Div(id="p2-slider-outputDivId-show", children=''),
                                                            html.Div(id="p2-slider-outputDivId-hide", children=''),

                                                            html.Div([
                                                                html.Button(id='p2-button-buttonId', n_clicks=0, children='Prepare dataset')]
                                                            ),


                                                            html.Div(id='p2-button-outputDivId-hide', style={'display': 'none'}),
                                                            html.H3(id='p2-button-outputDivId-show', children=''),
                                                            dcc.Interval(id='p2-interval2', interval=1 * 1000, n_intervals=0),

                                                            dcc.Textarea(id='p2-console-out',
                                                                placeholder='Enter a value...',
                                                                value='This is a TextArea component\nTesting',
                                                                style={'width': '100%'}
                                                        )
                                                        ])



                ]),

                html.Div(className="box-grid-container", children=[
                        html.Div(className="box box1", children=[dcc.Link('Go to Module 1', href='/page-1')]),
                        html.Div(className="box box2", children=[dcc.Link('Go to Module 2', href='/page-2')]),
                        html.Div(className="box box3", children=[dcc.Link('Go to Module 3', href='/page-3')]),
                        html.Div(className="box box3", children=[dcc.Link('Go to HOME', href='/index')])
                ])
            ])

])


