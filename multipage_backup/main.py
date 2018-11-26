import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import sys

import flask
from app_pages import app1, app2, app3, index
from app_pages import functions_nn as fnn
from app_pages import functions_frontend as fnf

#CSS
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Generate app and server
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True

#Output File
f = open('out.txt', 'w')
f.close()


url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

raw = pd.read_csv('https://raw.githubusercontent.com/kestefon/dev/master/data.csv')

def serve_layout():
    if flask.has_request_context():
        return url_bar_and_content_div
    return html.Div([
        url_bar_and_content_div,
        index.layout,
        app1.layout,
        app2.layout,
        app3.layout,
    ])


app.layout = serve_layout


# Index Page Callbacks
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == "/page-1":
        return app1.layout
    elif pathname == "/page-2":
        return app2.layout
    elif pathname == "/page-3":
        return app3.layout
    else:
        return index.layout


#Page 1 Callbacks
@app.callback(Output('p1-output-state', 'children'),
              [Input('p1-submit-button', 'n_clicks')],
               [State('p1-id-dropdown-data', 'value')])
def update_output(n_clicks, dropdown_value):
    return dropdown_value

@app.callback(Output('p1-intermediate-value', 'children'), [Input('p1-output-state', 'children')])
def clean_data(value):

     if value == "RAW":
         cleaned_df = raw
     elif value == "CLEAN":
         cleaned_df = fnn.data_cleanup(raw)


     # more generally, this line would be
     # json.dumps(cleaned_df)
     return cleaned_df.to_json(date_format='iso', orient='split')

@app.callback(Output('p1-table', 'children'), [Input('p1-intermediate-value', 'children')])
def update_table(jsonified_cleaned_data):
    dff = pd.read_json(jsonified_cleaned_data, orient='split')
    table = fnf.create_table(dff)
    return table



#PAGE 2 CALLBACKS Generate arrays & Test/Train split

@app.callback(Output('p2-output-state-arrays', 'children'),
              [Input('p2-id-button-arrays', 'n_clicks')])
def click_generate(n_clicks, value="GEN"):
    return value

@app.callback(Output('p2-div-out', 'children'),
              [Input("p2-output-state-arrays", 'children')])
def generate(value):
    if value=="GEN":

        orig_stdout = sys.stdout
        f = open('out.txt', 'a')
        sys.stdout = f
        print(fnn.generate_arrays(fnn.data_cleanup(raw))[0])
        sys.stdout = orig_stdout
        f.close()
        return "Generated Arrays"


@app.callback(dash.dependencies.Output('p2-console-out',
'srcDoc'),
    [dash.dependencies.Input('p2-interval2', 'n_intervals')])
def update_output(n):
    file = open('out.txt', 'r')
    data=''
    lines = file.readlines()
    if lines.__len__()<=20:
        last_lines=lines
    else:
        last_lines = lines[-20:]
    for line in last_lines:
        data=data+line + '<BR>'
    file.close()
    return data


@app.callback(Output(component_id = 'p2-id-div-split', component_property= 'children'),
              [Input('p2-id-slider-split', 'value')])
def show_split(value):
    return "Train: {} %".format(100-value) + "Test: {}%".format(value)





if __name__ == '__main__':
    app.run_server(debug=True)
