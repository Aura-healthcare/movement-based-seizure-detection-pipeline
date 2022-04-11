import dash
from .picker import *
from .graph import *

def render_window():
    return dash.html.Div(
        children=[
            dash.html.Div(
                children=[
                    dash.html.P(children="📈", className="header-emoji"),
                    dash.html.H1(
                        children="Seizure Viewer", className="header-title"
                    ),
                    dash.html.P(
                        children=["Visualize the medical data of epileptic ",
                                dash.html.Br(),
                                "patients between 2005 and 2010"],
                        className="header-description",
                    ),
                ],
                className="header",
            ), # ---------------Menus deroulants---------------
            dash.html.Div(
                children=[
                    render_picker('Patient', 'patient-filter'),
                    render_picker('Seizure', 'seizure-filter'),
                    render_picker('Eventfile', 'event-filter')
                ],
                className="menu"
            ), # ---------------ECG-Graph ---------------
            dash.html.Div(
                children=[
                    render_graph('ecg-graph'),
                    render_graph('ecg-graph-2'),
                    dash.html.Div(
                        children=[
                            dash.html.Img(id="graph3", className="image"),
                            dash.html.Img(id="graph4", className="image")
                        ],
                        className="container"
                    ),
                    dash.html.Div(
                        children=[
                            dash.dcc.Dropdown(
                                id="wavelets-filter",
                                value=None,
                                clearable=True,
                                multi=False,
                                searchable=True,
                                className="dropdown",
                            ),
                            render_graph('wavelets-graph'),
                            render_graph('graph5'),
                        ],
                        className="card",
                    ),
                    dash.html.Div(
                        children=[
                            render_graph('wavelets-graph-2'),
                            render_graph('graph6'),
                        ],
                        className="card",
                    )
                ],
                className="wrapper",
            ),
        ],
        className="mainview"
    );