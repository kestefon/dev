import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
import keras
from keras.preprocessing.text import Tokenizer
import sys
import numpy as np

# df1 = pd.read_csv(
#     'https://gist.githubusercontent.com/chriddyp/'
#     'c78bf172206ce24f77d6363a2d754b59/raw/'
#     'c353e8ef842413cae56ae3920b8fd78468aa4cb2/'
#     'usa-agricultural-exports-2011.csv')

raw = pd.read_csv('https://raw.githubusercontent.com/kestefon/dev/master/data.csv')

def replaceNone(r):
    if r == None:
        return "None"
    else:
        return str(r)

def data_cleanup(dataframe):
    data_split = raw.Sequence.str.split(pat="-", expand=True)
    cleaned_data = pd.concat([raw, data_split], axis=1)
    cleaned_data.drop(columns=["Sequence"], inplace=True)
    cleaned_data = pd.melt(cleaned_data, id_vars=["User ID"])
    cleaned_data.sort_values(["User ID", "variable"], inplace=True)

    # rename columns
    cleaned_data.rename(columns={'User ID': 'user_id', 'variable': 'timestep', 'value': 'event'}, inplace=True)

    #handle None values
    cleaned_data.reset_index(level=0, drop=True, inplace=True)
    cleaned_data['event'] = cleaned_data.event.apply(replaceNone)

    #tokenize
    tk = Tokenizer()
    tk.fit_on_texts(cleaned_data.event.values)
    word_index = tk.word_index
    word_index = word_index.copy()
    word_index.pop("none")

    enc = tk.texts_to_matrix(cleaned_data.event.values, mode="binary")
    enc = pd.DataFrame(enc)
    cleaned_data = pd.concat([cleaned_data, enc], axis=1)
    cleaned_data['next_event'] = cleaned_data['event'].shift(-1)

    def checkDict(event, next_event, timestep):
        if event.lower() in word_index and next_event.lower() not in word_index:
            return "Last Event"
        elif timestep == 6 and event.lower() in word_index:
            return "Last Event"
        else:
            return "-"

    cleaned_data.next_event.fillna("None", inplace=True)
    cleaned_data['flag'] = cleaned_data.apply(lambda x: checkDict(x['event'], x['next_event'], x['timestep']), axis=1)
    cleaned_data.drop(columns=[0, 1], inplace=True)

    return cleaned_data



def generate_arrays(dataframe):
    arr_x = []
    arr_y = []
    for name, group in dataframe.groupby('user_id'):
        grp = np.array(group)
        # print("full group:\n", grp)
        last = grp[grp[:, 8] == "Last"]
        # print("last:\n",last)
        keep = grp[grp[:, 8] != "Last"]
        # print("keep:\n",keep)
        arr_x.append(keep)
        arr_y.append(last)


    data_x = np.stack(arr_x, axis=0)
    data_x = data_x[:, :, 3:7]
    data_x = data_x.astype(int)

    data_y = np.stack(arr_y, axis=0)
    data_y = data_y[:, :, 3:7]
    data_y = data_y.astype(int)

    return([data_x, data_y])


def create_table(dataframe, max_rows=5):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


f = open('out.txt', 'w')
f.close()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Dropdown(id='id-dropdown-data',
        options=[
            {'label': 'Raw Data', 'value': 'RAW'},
            {'label': u'Cleaned Data', 'value': 'CLEAN'}
        ],
        value='RAW'
    ),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    html.Div([
        html.Button(id='id-button-arrays', n_clicks=0, children='Generate Arrays')]
    ),
    html.H4(children='Choose Train/Test % Split'),
    dcc.Slider(id='id-slider-split',
        min=15,
        max=50,
        marks={i: '{}%'.format(i) for i in range(15,51,5)},
        value=30,
        updatemode='drag'
    ),
    html.Div(id="id-div-split", children=''),
    # dcc.Textarea(id='id-textarea-split',
    #     placeholder='Choose test/train % using slider...',
    #     value='',
    #     style={'width': '50%'}
    # ),
    html.Button(id='id-button-split', n_clicks=0, children='Split Dataset'),
    html.Div(id='intermediate-value', style={'display': 'none'}),
    html.Div(id='output-state', style={'display': 'none'}),
    html.Div(id='output-state-arrays', style={'display': 'none'}),
    html.H4(children='Data Preview'),
    html.Table(id='table'),
    dcc.Interval(id='interval2', interval=60 * 1000, n_intervals=0),
    html.H1(id='div-out', children=''),
    html.Iframe(id='console-out',srcDoc='',style={'width':
'100%','height':200})
])
@app.callback(Output(component_id = 'id-div-split', component_property= 'children'),
              [Input('id-slider-split', 'value')])
def show_split(value):
    return "Train: {} %".format(100-value) + "Test: {}%".format(value)

@app.callback(Output('output-state', 'children'),
              [Input('submit-button', 'n_clicks')],
               [State('id-dropdown-data', 'value')])
def update_output(n_clicks, dropdown_value):
    return dropdown_value

@app.callback(Output('intermediate-value', 'children'), [Input('output-state', 'children')])
def clean_data(value):

     if value == "RAW":
         cleaned_df = raw
     elif value == "CLEAN":
         cleaned_df = data_cleanup(raw)


     # more generally, this line would be
     # json.dumps(cleaned_df)
     return cleaned_df.to_json(date_format='iso', orient='split')

@app.callback(Output('table', 'children'), [Input('intermediate-value', 'children')])
def update_table(jsonified_cleaned_data):
    dff = pd.read_json(jsonified_cleaned_data, orient='split')
    table = create_table(dff)
    return table



@app.callback(Output('output-state-arrays', 'children'),
              [Input('id-button-arrays', 'n_clicks')])
def click_generate(n_clicks, value="GEN"):
    return value

@app.callback(Output('div-out', 'children'),
              [Input("output-state-arrays", 'children')])
def generate(value):
    if value=="GEN":

        orig_stdout = sys.stdout
        f = open('out.txt', 'a')
        sys.stdout = f
        print(generate_arrays(data_cleanup(raw))[0])
        sys.stdout = orig_stdout
        f.close()
        return "Generated Arrays"


@app.callback(dash.dependencies.Output('console-out',
'srcDoc'),
    [dash.dependencies.Input('interval2', 'n_intervals')])
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

if __name__ == '__main__':
    app.run_server(debug=True)


