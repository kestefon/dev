import dash
import dash_core_components as dcc
import dash_html_components as html
from .templates import master_template as tp

html.Br()

home_content = \
    html.Div(id="main-container-home", className="main-container", children=[

        html.Div(className="home-bg", children=[
            html.Button(id='next-button-1', className="hvr-fade", children=[
                dcc.Link('Next Page', href='/page-1', className="link-button")
            ]),
            html.H3(children="Intro: Predicting Consumer Behavior with Log Data", className="page-header-text"),
            html.Hr(),
            html.Div(className="home-text", children=[

html.H5("Business Challenge:", className="subheader-1"),
html.Ul([
html.Li("Can we predict what action a user will take next on the site?"),
html.Li("Can we predict which creative a user will click next?"),
html.Li("How should we sequence our creatives for a sequential messaging campaign?"),
], className="subtext"),

html.H5("Statistical Translation:", className="subheader-1"),
html.Ul([
html.Li("What is the probability of [event], given [sequence of events that preceded it]?.")
], className="subtext"),
html.Hr(),
html.H5("Common Approaches to Sequence Prediction", className="subheader-1"),

html.H6("Variable-Order Markov Models", className="subheader-2"),
html.Ul([
html.Li("In a nutshell: This is a probability model, "
        "in which observed conditional probabilities are used to determine the most likely next action, "
        "given all previous actions", className="subtext")
]),
html.H6("Recurrent Neural Networks", className="subheader-2"),
html.Ul([
html.Li("In a nutshell: This is a machine learning approach to the same problem, "
        "in which a model learns to predict a target "
        "(next action) based on an input "
        "(all previous actions). "
        "At their core, neural networks work by predicting a target, evaluating performance,"
        " and then updating weights algorithmically in order to improve the prediction.")
], className="subtext")




            ])

        ])

    ])

layout = tp.create_html_template(home_content, outer_div_id="outer-div-home")