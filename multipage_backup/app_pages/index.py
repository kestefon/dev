import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State




layout = html.Div([
    dcc.Link('Navigate to "/page-1"', href='/page-1'),
    html.Br(),
    dcc.Link('Navigate to "/page-2"', href='/page-2'),
])






