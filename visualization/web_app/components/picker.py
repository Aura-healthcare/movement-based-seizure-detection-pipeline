import dash

def render_picker(children, id_name):
    return dash.html.Div(
		children=[
			dash.html.Div(children=children, className="menu-title"),
			dash.dcc.Dropdown(
				id=id_name,
				value=None,
				clearable=True,
				multi=False,
				searchable=True,
			    className="dropdown",
			)
		]
	);