import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

from files import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# df = pd.read_csv(
#     'https://gist.githubusercontent.com/chriddyp/' +
#     '5d1ea79569ed194d432e56108a04d188/raw/' +
#     'a9f9e8076b837d541398e999dcbac2b2826a81f8/'+
#     'gdp-life-exp-2007.csv')

df = pd.read_csv('/home/stefon/PersonalGit/dev/plot_test/seq_df.csv')
test_set=df['datatype']=='test'
df_test=df[test_set]
df_train=df[~test_set]

x_epoch=df.epoch.unique()

trace_test = go.Scatter(
    x=x_epoch,
    y=df_test['error'],
    name = "Test Loss",
    line = dict(color = '#17BECF'),
    opacity = 0.8)

trace_train = go.Scatter(
    x=x_epoch,
    y=df_train['error'],
    name = "Train Loss",
    line = dict(color = '#7F7F7F'),
    opacity = 0.8)

data = [trace_test, trace_train]



app.layout = html.Div([
    dcc.Graph(
        id='life-exp-vs-gdp',
        figure={
            'data': data



        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)