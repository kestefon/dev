import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer

from .templates import master_template as tp

p2_content= \
    html.Div(id="main-container-p2", className="main-container", children=[



        html.Div(id="p2-content-box", children=[
        tp.page_title(html.H3(children="Module 2")),

        tp.page_item(html.H4(children='Choose Train/Test % Split')),
        tp.page_item(dcc.Slider(id='p2-slider-sliderId',
                   min=15,
                   max=50,
                   marks={i: '{}%'.format(i) for i in range(15, 51, 5)},
                   value=30,
                   updatemode='drag'
                   )),

        tp.page_item(html.Div(id="p2-slider-outputDivId-show", children='')),


        tp.page_item(
            html.Button(id='p2-button-buttonId', n_clicks=0, children='Prepare dataset')
        ),
        tp.page_item(html.H3(id='p2-button-outputDivId-show', children='')),
        tp.page_item(dcc.Textarea(id='p2-console-out',
                     placeholder='Enter a value...',
                     value='This is a TextArea component\nTesting',
                     style={'width': '100%'}
                     ))
        ]),


        dcc.Interval(id='p2-interval2', interval=1 * 1000, n_intervals=0),
        html.Div(id="p2-slider-outputDivId-hide", children='', style={'display': 'none'}),
        html.Div(id='p2-button-outputDivId-hide', style={'display': 'none'})


])

layout = tp.create_html_template(p2_content)