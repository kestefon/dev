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

        tp.page_item(content=html.Div([
            html.Div(className='slider-item', children='Choose Train/Test % Split'),
            html.Div(className='slider-item', children=[dcc.Slider(id='p2-slider-sliderId',
                       min=20,
                       max=40,
                       marks={i: '{}%'.format(i) for i in range(20, 41, 10)},
                       value=30,
                       updatemode='drag'
                       )]),

        html.Div(id="p2-slider-outputDivId-show", className='slider-item', children='')]), make_id='slider-unit'),


        tp.page_item(
            html.Button(id='p2-button-buttonId', n_clicks=0, children='Prepare dataset')
        ),

        dcc.Interval(id='interval1', interval=1*500, n_intervals=0),
        dcc.Interval(id='interval2', interval=1 *500, n_intervals=0),
        tp.page_item(html.Div(id='console-out',children='')),



        tp.page_item(html.Img(id='cur_plot', src='', style={'width': '100%', 'max-width': '600px'}))
        ]),


        html.Div(id="p2-slider-outputDivId-hide", children='', style={'display': 'none'}),
        html.Div(id='p2-button-outputDivId-hide', style={'display': 'none'}),
        html.Div(id='p2-hidden-div-1', children='', style={'display': 'none'})


])

layout = tp.create_html_template(p2_content)