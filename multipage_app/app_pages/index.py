import dash
import dash_core_components as dcc
import dash_html_components as html
from .templates import master_template as tp


html.Br()

home_content=\
    html.Div(id= "main-container-home", className="main-container", children=[

        html.Div(className="home-bg", children=[
            html.H3(children="Intro: Optimal Sequencing of Ads With Neural Networks"),
            html.Div(className="home-text", children=[

              html.P([

              html.P([
                    html.H5('Background:'),
                    html.P('If we know that a user clicked Ad X, can we predict which ad they are most likely to click next?'),
                ], className="p-group"),

            html.P([
                html.H5('Approach:'),
                html.P('1) Compile & pre-process log-level dataset, which details each ad a user has clicked.'),
                html.P('2) Train a "recurrent neural network:'),
                html.P('Separate dataset into "train" and "test" sets.', className="p-indent"),
                html.P(children=['Fit model using training set only.'],className="p-indent"),
                html.P('Use model to predict probability of clicking NEXT AD, given that a user has seen PREVIOUS ADS.', className="p-indent"),
                html.P('Compare predictions to actual next ad clicked using "test" set.', className="p-indent")
                ], className='p-group')
         ], className='main-p')


            
            
            ])

        ])


    ])

layout = tp.create_html_template(home_content, outer_div_id="outer-div-home")