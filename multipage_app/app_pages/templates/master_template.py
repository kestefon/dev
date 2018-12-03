import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.graph_objs as go


def create_html_template(content, outer_div_id=None):
    master_layout = \
        html.Div(className = "outer-div", id=outer_div_id, children=[
            html.Div(className="left-div", children=[content]),
            html.Div(className="right-div", children=[
                html.Div(className="box-container", children=[
                    html.Div(className="box box1 hvr-shrink", children=[
                        html.H3(className="display-text", children=["Sequence Data Preview"]),
                        dcc.Link('Go to Module 1', href='/page-1', className="link")
                    ]),
                    html.Div(className="box box2 hvr-shrink", children=[
                        html.H3(className="display-text", children=["Train Neural Network"]),
                        dcc.Link('Go to Module 2', href='/page-2', className="link")

                    ]),
                    # html.Div(className="box boxnone", children=[
                    #
                    # ]),
                    html.Div(className="box boxhome hvr-shrink", children=[
                        html.H3(className="display-text", children=["Home"]),
                        dcc.Link('Return to Intro', href='/index', className="link")
                    ])
                ])
            ])


        ])
    return master_layout


def page_item(content, make_id=None):
    if make_id is None:
        page_item = html.Div(className="page-item", children=content)
    else:
        page_item = html.Div(id=make_id, className="page-item", children=content)
    return page_item


def page_title(content):
    page_title = html.Div(className="page-title", children=[content])
    return page_title