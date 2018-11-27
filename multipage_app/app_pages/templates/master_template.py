import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer



def create_html_template(content):
    master_layout = \
        html.Div(className = "layout_container",
            children=[html.Div(className='menu-container',
                children=[html.Div(className="menu",
                    children=[
                        html.Div(className="date", children=["Aug 14, 2016"]),
                        html.Div(className="links",
                                 children=[
                                    html.Div(className="signup",children=["Sign Up"]),
                                    html.Div(className="login",children=["Login"])
                                          ])
                            ])
                         ]),

            html.Div(className="main-page", children=[content,

                html.Div(className="box-grid-container", children=[
                    html.Div(className = "overlay-container", children = [
                        html.Div(className="box box1", children=[
                            html.H3(className="display-text", children=["Data Preview"]),
                            dcc.Link('Go to Module 1', href='/page-1', className="link")
                        ])

                    ]),
                html.Div(className="overlay-container", children=[
                        html.Div(className="box box2", children=[
                            html.H3(className="display-text", children=["Build Neural Net"]),
                            dcc.Link('Go to Module 2', href='/page-2', className="link")
                        ])
                        ]),
                html.Div(className="overlay-container", children=[
                        html.Div(className="box box3", children=[
                            html.H3(className="display-text", children=["Evaluate Neural Net"]),
                            dcc.Link('Go to Module 3', href='/page-3', className="link")
                        ])
                        ]),
                html.Div(className="overlay-container", children=[
                        html.Div(className="box box_home", children=[
                            html.H3(className="display-text", children=["HOME"]),
                            dcc.Link('Return to Home', href='/index', className="link")
                        ])
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

def page_group(*args):
    page_group = html.Div(className="page-group", children=[args])
    return page_group

def page_title(content):
    page_title = html.Div(className="page-title", children=[content])
    return page_title
