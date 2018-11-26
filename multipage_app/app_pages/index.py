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
                html.Div(className="main-container", children=["main container"]),

                html.Div(className="box-grid-container", children=[
                        html.Div(className="box box1", children=[dcc.Link('Go to Module 1', href='/page-1')]),
                        html.Div(className="box box2", children=[dcc.Link('Go to Module 2', href='/page-2')]),
                        html.Div(className="box box3", children=[dcc.Link('Go to Module 3', href='/page-3')]),
                        html.Div(className="box box3", children=[dcc.Link('Go to HOME', href='/index')]),
                ])
            ])

])

