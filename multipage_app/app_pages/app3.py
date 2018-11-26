import dash
import dash_core_components as dcc
import dash_html_components as html


layout = html.Div(className = "layout_container", children=[html.Div(
    className='menu-container',
                  children=[
                      html.Div(className="menu", children=
                                     [html.Div(className="date",children=["Aug 14, 2016"]),
                                      html.Div(className="links",children=[
                                          html.Div(className="signup",children=["Sign Up"]),
                                          html.Div(className="login",children=["Login"])
                                      ])

                                    ])
                  ]),

            html.Div(className="main-page", children=[
                html.Div(id= "main-container-p3", className="main-container", children=["main container"]),

                html.Div(className="box-grid-container", children=[
                        html.Div(className="box box1", children=[
                            html.H3(className="display-text", children=["Data Preview"]),
                            dcc.Link('Go to Module 1', href='/page-1', className="link")]),
                        html.Div(className="box box2", children=[
                            html.H3(className="display-text", children=["Build Neural Net"]),
                            dcc.Link('Go to Module 2', href='/page-2', className="link")]),

                        html.Div(className="box box_home", children=[
                            html.H3(className="display-text", children=["HOME"]),
                            dcc.Link('Return to Home', href='/index', className="link")]),

                        html.Div(className="box box3", children=[
                            html.H3(className="display-text", children=["Evaluate Neural Net"]),
                            dcc.Link('Go to Module 3', href='/page-3', className="link")])
                ])
            ])

])

