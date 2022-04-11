import dash

def render_graph(id_name):
    return dash.html.Div(
		children=dash.dcc.Graph(
		    id=id_name,
		    config={ "displayModeBar": False },
		    className="graph"
		)
	);