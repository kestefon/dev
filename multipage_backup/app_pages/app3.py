import dash
import dash_core_components as dcc
import dash_html_components as html


layout = html.Div([
    html.H3(children="App 3"),
    dcc.Link('Go to App 1', href='/page-1'),
    dcc.Link('Go to App 2', href='/page-2'),
    dcc.Link('Go to App 3', href='/page-3')
])


