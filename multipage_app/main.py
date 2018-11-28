import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import sys

import flask
from app_pages import app1, app2, index, collapse
from app_pages import functions_nn as fnn
from app_pages import functions_frontend as fnf
from app_pages import rnn_sequence

from plotnine import *
from io import BytesIO
import base64
import os
import plotly.graph_objs as go


#os.remove('out.txt')


#create tempfile for stdout

f = open('out.txt', 'w+')
f.close()

#CSS
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


#Generate app and server
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True



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
        app2.layout
    ])




app.layout = serve_layout


#GENERAL FUNCTIONS

def gen_layer_list(inputs=[{'layer': 1, 'type': 'lstm', 'n_hidden': 32}]):
    l1 = {'layer': 1, 'type': 'lstm', 'n_hidden': 32}
    l2 = {'layer': 2, 'type': 'gru', 'n_hidden': 32}
    layer_list = []
    layer_list.append(l1)
    layer_list.append(l2)
    return layer_list


def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.save(out_img, format='png', **save_args)
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    out_label="data:image/png;base64,{}".format(encoded)
    return out_label

# Index Page Callbacks
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == "/page-1":
        return app1.layout
    elif pathname == "/page-2":
        return app2.layout
    elif pathname == "/index":
        return index.layout
    else:
        return index.layout


#Page 1 Callbacks
# @app.callback(Output('p1-output-state', 'children'),
#               # [Input('p1-submit-button', 'n_clicks')],
#                [Input('p1-id-dropdown-data', 'value')]
#               )
# def update_p1_output(dropdown_value):
#     return dropdown_value

@app.callback(Output('p1-intermediate-value', 'children'), [Input('p1-id-dropdown-data', 'value')])
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

@app.callback(Output(component_id = 'p2-testtrain-show', component_property= 'children'),
              [Input('p2-slider-sliderId', 'value')])
def slider_output_show(value):
    return "Train Dataset: {}% ".format(100-value) + "Test Dataset: {}%".format(value)

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




def df_to_plotly(df):
    test_set = df['datatype'] == 'test'
    df_test = df[test_set]
    df_train = df[~test_set]

    x_epoch = df.epoch.unique()

    trace_test = go.Scatter(
        x=x_epoch,
        y=df_test['error'],
        name="Test Loss",
        line=dict(color='#17BECF'),
        opacity=0.8)

    trace_train = go.Scatter(
        x=x_epoch,
        y=df_train['error'],
        name="Train Loss",
        line=dict(color='#7F7F7F'),
        opacity=0.8)

    data = [trace_test, trace_train]
    return {
        'data': data
    }


@app.callback(Output('cur_plot', 'figure'),
              [Input('p2-button-buttonId', 'n_clicks')],
               [State('p2-slider-sliderId', 'value'),
                State('choose_cell-1', 'value'),
                State('choose_cell-2', 'value'),
                State('choose_cell-3', 'value'),
                State('choose_cell-4', 'value'),
                State('layer-value-1', 'value'),
                State('layer-value-2', 'value'),
                State('layer-value-3', 'value'),
                State('layer-value-4', 'value'),
                State('p2-epoch-hide', 'value')
                ])

def update_plot(n_clicks, test_percent, cell1=None,cell2=None,cell3=None,cell4=None,
                layer1=None,layer2=None,layer3=None,layer4=None, n_epochs=None):

    if n_epochs is None:
        n_ep=100
    else:
        n_ep=n_epochs

    layer_list = []
    l1 = {'layer': 1, 'type': 'lstm', 'n_hidden': 32}
    layer_list.append(l1)

    l2 = {'layer': 2, 'type': 'gru', 'n_hidden': 32}
    layer_list.append(l2)

    l3 = {'layer': 3, 'type': 'gru', 'n_hidden': 32}
    layer_list.append(l3)

    l4 = {'layer': 4, 'type': 'gru', 'n_hidden': 32}
    layer_list.append(l4)


    if n_clicks % 2 == 1:
        orig_stdout = sys.stdout
        f = open('out.txt', 'a')
        sys.stdout = f
        print("Running model...")

        data_list_temp = fnn.generate_arrays(fnn.data_cleanup(raw), var_test_percent=int(test_percent)/100)

        model = fnn.fit_rnn(data_inputs=data_list_temp,
                                  num_epochs=int(n_ep), batch_size=25, learning_rate=0.003, layers=layer_list)
        # out_plot = (ggplot(data=model['history_df']) +
        #             geom_point(mapping=aes(x="epoch", y="error", color="datatype")) +
        #             xlab("Epoch") + ylab("Error") + labs(color="Data"))
        sys.stdout = orig_stdout
        f.close()
        return df_to_plotly(model['history_df'])


@app.callback(dash.dependencies.Output('console-out',
'srcDoc'),
    [dash.dependencies.Input('interval2', 'n_intervals')])
def update_output(n):
    data = ''
    file = open('out.txt', 'r')
    lines = file.readlines()
    if lines.__len__()<=5:
        last_lines=lines
    else:
        last_lines = lines[-5:]
    for line in last_lines:
        data=data+line + '<BR>'
    file.close()
    return data


@app.callback(
    dash.dependencies.Output('button_container', 'style'),
    [dash.dependencies.Input('button', 'n_clicks')],
    )
def button_toggle(n_clicks):
    if n_clicks % 2 == 1:
        return {'display': 'none'}
    else:
        return {'display': 'block'}




if __name__ == '__main__':
    app.run_server(debug=True)
