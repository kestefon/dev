import dash
import dash_core_components as dcc
import dash_html_components as html
from .templates import master_template as tp




home_content=\
    html.Div(id= "main-container-home", className="main-container", children=["main container"])

layout = tp.create_html_template(home_content)