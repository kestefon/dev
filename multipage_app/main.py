import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event
import pandas as pd
# from flask_caching import Cache
import numpy as np
from keras.preprocessing.text import Tokenizer
import sys
import plotly.graph_objs as go
import plotly.plotly as py
import tempfile
import gunicorn
import json
# from dateutil import tz
#
# #Time Zone Conversion
# from_zone = tz.tzutc()
# to_zone = tz.tzlocal()



import flask
from app_pages import app1, app2, index, collapse
from app_pages import functions_nn as fnn

import datetime
import time

from plotnine import *
from io import BytesIO
import base64
import os


from rq import Queue
from worker import conn

q = Queue('high', is_async=False, connection=conn, default_timeout=60*3)

raw = pd.read_csv('https://raw.githubusercontent.com/kestefon/dev/master/data.csv')
data_handler=fnn.DataHandler(raw)
out_data=data_handler.data_cleanup()

#learning rate factor


#CSS
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                        {
                            'href': 'https://fonts.googleapis.com/css?family=Roboto:100,200,300,400,500,600,700,900',
                            'ref':'stylesheet'
                        },
                        {
                            'href': 'css/hover.css',
                            'ref': 'stylesheet',
                            'media':'all'
                        }

                        ]





semaphore = fnn.Semaphore()




#Generate app and server
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True

#
# cache = Cache(app.server, config={
#     'CACHE_TYPE': 'redis',
#     'CACHE_REDIS_URL': os.environ.get('REDIS_URL', '')
# })
#
# TIMEOUT = 60
# @cache.memoize(timeout=TIMEOUT)
#






url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])



def serve_layout():
    if flask.has_request_context():
        return url_bar_and_content_div
    return html.Div([
        url_bar_and_content_div,
        index.layout,
        app1.layout,
        app2.layout
    ])




app.layout = serve_layout



# Index Page Callbacks
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == "/index":
        return index.layout
    elif pathname == "/page-1":
        return app1.layout
    elif pathname == "/page-2":
        return app2.layout
    else:
        return index.layout


#Page 1 Callbacks
# @app.callback(Output('p1-output-state', 'children'),
#               # [Input('p1-submit-button', 'n_clicks')],
#                [Input('p1-top-dropdown', 'value')]
#               )
# def update_p1_output(dropdown_value):
#     return dropdown_value

@app.callback(
    Output('lock', 'options'),
    [Input('interval_counter', 'n_intervals')],
    events=[Event('interval', 'interval')])
def display_status_counter(counter):
    if semaphore.is_locked() == True:

        #now_local=datetime.datetime.now().replace(tzinfo=from_zone).strftime("%Y-%b-%d: %H:%M:%S")
        now_local = datetime.datetime.now().strftime("%Y-%m-%d: %H:%M:%S")
        return {'label': 'Running... Current time: ' + now_local, 'value': 'Running...'}, \
            {'label': 'Free', 'value': 'Free'}
    else:
        return {'label': 'Running...', 'value': 'Running...'}, \
               {'label': 'Free', 'value': 'Free'}


@app.callback(
    Output('lock', 'value'),
    events=[Event('interval', 'interval')])
def display_status():
    return 'Running...' if semaphore.is_locked() else 'Free'



@app.callback(Output('p1-intermediate-value', 'children'), [Input('p1-top-dropdown', 'value')])
def clean_data(value):

     if value == "RAW":
         cleaned_df = out_data['raw']
     elif value == "CLEAN":
         cleaned_df = out_data['clean'][0]
     else:
         cleaned_df = out_data['raw']



     # more generally, this line would be
     # json.dumps(cleaned_df)
     return cleaned_df.to_json(date_format='iso', orient='split')

@app.callback(Output('p1-table', 'children'), [Input('p1-intermediate-value', 'children')])
def update_table(jsonified_cleaned_data):
    if jsonified_cleaned_data is not None:
        dff = pd.read_json(jsonified_cleaned_data, orient='split')
        table = fnn.create_table(dff)
        #table = q.enqueue(fnn.create_table, dff)
        print(table)

        return table
    else:
        return None


@app.callback(Output('p1-toggle-table', 'style'), [Input('p1-top-dropdown', 'value')])

def toggle_table(value):
    print("toggle_table function start", value)
    if value != "TENSOR":
        print("value !=TENSOR", value)
        return {'display': 'block'}
    else:
        print("else...", value)
        return {'display': 'none'}


@app.callback(Output('p1-toggle-graph', 'style'), [Input('p1-top-dropdown', 'value')])
def toggle_graph(value):
    if value == "TENSOR":
        return {'display': 'block'}
    else:
        return {'display': 'none'}




#PAGE 2 CALLBACKS Generate arrays & Batch size

@app.callback(Output(component_id = 'p2-batchsize-show', component_property= 'children'),
              [Input('p2-slider-sliderId', 'value')])
def slider_output_show(value):
    return "Batch Size: {} ".format(value)

@app.callback(Output(component_id = 'p2-slider-outputDivId-hide', component_property= 'children'),
              [Input('p2-slider-sliderId', 'value')])
def slider_output_hide(value):
    return value



@app.callback(Output(component_id = 'p2-epoch-show', component_property= 'children'),
              [Input('p2-epoch-slider', 'value')])
def slider_output_show(value):
    return html.Div([html.P("Model will run for "), html.Span("{}".format(value)),
                            html.P("epochs. Results will be displayed in plot below.")])

@app.callback(Output(component_id = 'p2-epoch-hide', component_property= 'children'),
              [Input('p2-epoch-slider', 'value')])
def slider_output_hide(value):
    return value


#COMPONENTS OF PAGE
#Slider
#Button
#Iframe





@app.callback(Output('batch-size-value','children'),
              [Input('p2-slider-sliderId', 'value')])

def percent_val(value):
    return html.P("{}".format(value))

@app.callback(Output('lr-value','children'),
              [Input('p2-lr', 'value')])

def lr_val(value):
    return html.P("{}".format(round(value/app2.lrf,3)))



@app.callback(Output('epoch-value','children'),
              [Input('p2-epoch-slider', 'value')])

def epoch_val(value):
    return html.P("{}".format(value))


@app.callback(Output('submit-description','children'),
              [Input('p2-slider-sliderId', 'value'),
               Input('p2-lr', 'value'),
               Input('p2-epoch-slider', 'value'),
               Input('choose-cell-1', 'value'),
               Input('choose-cell-2', 'value'),
               Input('choose-cell-3', 'value'),
               Input('layer-value-1', 'value'),
               Input('layer-value-2', 'value'),
               Input('layer-value-3', 'value'),
               Input('layer-value-1-drop', 'value'),
                Input('layer-value-2-drop', 'value'),
                Input('layer-value-3-drop', 'value')

               ])

def submission_description(batch=None, lr=None, epoch=None, cell1=None, cell2=None, cell3=None,
                           layer1=None, layer2=None, layer3=None, drop1=None, drop2=None, drop3=None):

    if drop1 is None:
        drop1=1
    if drop2 is None:
        drop2=1
    if drop3 is None:
        drop3=1

    if not batch:
        batch_out = ''
    else:
        batch_out = html.P("Batch Size: {}".format(batch))

    if not lr:
        lr_out = ''
    else:
        lr_out = html.P("Learning Rate: {}".format(lr/app2.lrf))

    if not epoch:
        epoch_out = ''
    else:
        epoch_out = html.P("Number of Epochs {}:".format(epoch))

    if not layer1:
        l1_out = ''
    else:
        l1_out = html.P("Layer 1 is {} with {} hidden units and {}% keep probability".format(cell1.upper(), layer1, drop1*100))

    if not layer2:
        l2_out = ''
    else:
        l2_out = html.P("Layer 2 is {} with {} hidden units and {}% keep probability".format(cell2.upper(), layer2, drop2*100))

    if not layer3:
        l3_out = ''
    else:
        l3_out = html.P("Layer 3 is {} with {} hidden units and {}% keep probability".format(cell3.upper(), layer3, drop3*100))


    output = html.Div([batch_out, lr_out, epoch_out, l1_out, l2_out, l3_out
        ])

    return output






@app.callback(Output('hidden-plot', "children"),state=
              [State('p2-slider-sliderId', 'value'),
               State('p2-lr', 'value'),
               State('p2-epoch-slider', 'value'),
               State('choose-cell-1', 'value'),
               State('choose-cell-2', 'value'),
               State('choose-cell-3', 'value'),
               State('layer-value-1', 'value'),
               State('layer-value-2', 'value'),
               State('layer-value-3', 'value'),
              State('layer-value-1-drop', 'value'),
              State('layer-value-2-drop', 'value'),
              State('layer-value-3-drop', 'value')],
               events=[Event('button', 'click')])


def update_plot(batch, learning_rate, epoch,
                cell1, cell2, cell3, layer1, layer2, layer3, drop1, drop2, drop3):

    summary="Model submitted. Model settings are:"
    if not batch:
        batch_out = ''
    else:
        batch_out = html.P("Batch Size: {}".format(batch))

    if not learning_rate:
        lr_out = ''
    else:
        lr_out = html.P("Learning Rate: {}".format(learning_rate/app2.lrf))

    if not epoch:
        epoch_out = ''
    else:
        epoch_out = html.P("Number of Epochs {}:".format(epoch))

    if not layer1:
        l1_out = ''
    else:
        l1_out = html.P("Layer 1 is {} with {} hidden units".format(cell1, layer1))

    if not layer2:
        l2_out = ''
    else:
        l2_out = html.P("Layer 2 is {} with {} hidden units".format(cell2, layer2))

    if not layer3:
        l3_out = ''
    else:
        l3_out = html.P("Layer 3 is {} with {} hidden units".format(cell3, layer3))


    settings = html.Div([summary, batch_out, lr_out, epoch_out, l1_out, l2_out, l3_out
        ])
    d1={'layer': 1, 'type': cell1, 'n_hidden':layer1, 'rdrop': drop1}
    d2={'layer': 2, 'type': cell2, 'n_hidden':layer2, 'rdrop': drop2}
    d3={'layer': 3, 'type': cell3, 'n_hidden':layer3, 'rdrop': drop3}
    # print("printing layer/cell inputs...")
    # print(d1, d2, d3)
    # # d1={'layer': 1, 'type': cell1, 'n_hidden':32}
    # # d2={'layer': 2, 'type': cell2, 'n_hidden':32}
    # # d3={'layer': 3, 'type': cell3, 'n_hidden':32}
    layer_list=[d1,d2,d3]
    layer_list_copy=[]

    for i, d in enumerate(layer_list):
        print("i", i, d['n_hidden'])
        print('dnhidden = ', d['n_hidden'])
        print("rdrop", d['rdrop'])
        if d['rdrop'] is None:
            d['rdrop']=1
        if not d['n_hidden']:
            continue
        elif d['n_hidden']==0:
            continue
        else:
            print("keep record")
            layer_list_copy.append(layer_list[i])
            print("layer list copy", layer_list_copy)

    print("layer/cell inputs after removing nulls")
    print(layer_list_copy)
    print("running model...")



    long_output = q.enqueue(fnn.long_process, out_data['tensors'], num_epochs=epoch,
                            learning_rate=learning_rate/app2.lrf, layers=layer_list_copy, batch_size=batch)
    print("type", type(long_output.result))

    out=long_output.result
    print("successfully ran model--outputting plot")
    out_json=out.to_json(date_format='iso', orient='split')
    print(out_json)
    return out_json


@app.callback(Output('cur_plot', "figure"),
              [Input('hidden-plot', 'children')])

def reveal_plot(df_json):
    print("beginning of reveal_plot function")
    if df_json is not None:
        plot=pd.read_json(df_json, orient='split')
        print("updating plot")
        return fnn.df_to_plotly(plot)
    else:
        return fnn.default_plotly()

@app.callback(
    dash.dependencies.Output('layer-container-2', 'style'),
    [dash.dependencies.Input('layer-dropdown', 'value')],
    )
def reveal_layer2(value):
    if "L2" in value:
        return {'display': 'flex'}
    else:
        return {'display': 'none'}

@app.callback(
    dash.dependencies.Output('layer-container-3', 'style'),
    [dash.dependencies.Input('layer-dropdown', 'value')],
    )
def reveal_layer3(value):
    if "L3" in value:
        return {'display': 'flex'}
    else:
        return {'display': 'none'}


# app.scripts.config.serve_locally = True

if __name__ == '__main__':
    app.run_server(debug=True, threaded= True, processes=1)