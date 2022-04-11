import pathlib
import plotly
import numpy
import dash
import pywt
import sys
import os
import pandas as pd
from components import window
from pydoc import classname
from app_functions import *

sys.path.insert(0, '../..')
from classes.evt import *
from utils import utils

#########
# Model #
#########


data = pd.read_csv(pathlib.Path(SIMPLIFIED_EVT_PATH), index_col = 0)
liste_sz_model = [f[:-4] + '.evt' for f in os.listdir(DIR_TAGGED_EVT)]
data = data[data.filename.isin(liste_sz_model)]

stats_file_path = '../../data/seizure_statistics/Edf_acceleration_statistics.csv'
#data_stats = pd.read_csv(stats_file_path)
#data_stats = data_stats[(data_stats.mean_MSG < 1.03) & (data_stats.mean_MSG > 0.97)]
#liste_sz_model = list(data_stats.evt_name.unique())
#data = data[data.filename.isin(liste_sz_model)] 

list_interesting_items = ['ATF','AUTO','CLONIC','DYSTONIC','HYPER','HYPERTONIC',
'MYOCLONIC','NOMVT','OTHERS','SPASM','TONIC','VERSIF','TC']

data['nb_seizure'] = 0

data = data[data.STY.isin(list_interesting_items)]
list_patient = data.patient.unique()

for patient in list_patient:
    for sz in list_interesting_items:
        mask = (
		((data.patient == patient) | (data.patient == None)) 
		& ((data.STY == sz) | (data.STY == None)) 
	    )
        df_loc = data.loc[mask,:]
        nb_evt = df_loc.filename.nunique()
        data.loc[mask, 'nb_seizure'] = nb_evt

list_continuous_wavelet = pywt.wavelist(kind='continuous') 
list_discret_wavelet = pywt.wavelist(kind='discrete')

data["STI"] = pd.to_datetime(data["STI"],unit='ms')
data["ETI"] = pd.to_datetime(data["ETI"],unit='ms')


########
# View #
########

app = dash.Dash(__name__)
app.title = "Seizure Viewer"
app.layout = window.render_window()

##############
# Controller #
##############
'''  
option c'est pour avoir toutes les valeurs, sinon je mets value
Ici c'est la maj des menus déroulants
'''
@app.callback(
	dash.dependencies.Output("patient-filter", "options"),
	dash.dependencies.Output("seizure-filter", "options"),
	dash.dependencies.Output("event-filter", "options"),

	dash.dependencies.Input("patient-filter", "value"),
	dash.dependencies.Input("seizure-filter", "value"),
	dash.dependencies.Input("event-filter", "value"),

)
def update_widgets(patient, seizure, eventfile):
	

	patient_options=[
						{"label": p, "value": p}
						for p in numpy.sort(data.patient.unique())
					]
	seizure_options=[
						 {"label": s, "value": s}
						for s in numpy.sort(data.STY.unique())
					]
	eventfile_options=[
						{"label": e, "value": e}
						for e in numpy.sort(data.filename.unique())
					]

	if patient != None:
		seizure_options=[
						{"label": s, "value": s}
						for s in numpy.sort(data.loc[data.patient == patient].STY.unique())
						]	
		eventfile_options=[
						{"label": e, "value": e}
						for e in numpy.sort(data.loc[data.patient == patient].filename.unique())
						]
				
	if seizure != None:
		patient_options=[
						{"label": p, "value": p}
						for p in numpy.sort(data.loc[data.STY == seizure].patient.unique())
						]
		eventfile_options=[
						{"label": e, "value": e}
						for e in numpy.sort(data.loc[data.STY == seizure].filename.unique())
						]
	if eventfile != None:
		patient_options=[
						{"label": p, "value": p}
						for p in numpy.sort(data.loc[data.filename == eventfile].patient.unique())
						]
		seizure_options=[
						{"label": s, "value": s}
						for s in numpy.sort(data.loc[data.filename == eventfile].STY.unique())
						]					

	return patient_options, seizure_options, eventfile_options

'''  
Initialisation et MAJ du filtre wavelet, le fait de mettre options en output ça fait qu'il renvoit
toutes les valeurs possibles. A l'initialisatin la valeur c'est none. Les options sont les mêmes quel que
soit la valeur choisie.  
'''

@app.callback(
	dash.dependencies.Output("wavelets-filter", "options"),
	dash.dependencies.Input("wavelets-filter", "value"),
) 
def update_wavelets_filter(wavelet_type):
	'''
	vu qu'à la base notre menu déroulabnt est vide, on lui
	dit de se remplir. 
	Après ici quel que soit la valeur de value, on remplie toujurs la liste 
	avec toutes les valeurs possibles
	
	'''
	wavelets_options=[
						{"label": p, "value": p}
						for p in list_discret_wavelet 
					]
	print(' toto = ', wavelet_type)

	return wavelets_options
'''
callback maj graph ondelette
'''
@app.callback(
	dash.dependencies.Output("wavelets-graph", "figure"),
    dash.dependencies.Output("wavelets-graph-2", "figure"),
	dash.dependencies.Input("event-filter", "value"),
	dash.dependencies.Input("wavelets-filter", "value"),
)

def update_wavelets_graph(evt_csv_name,wavelets_type):
	
	print(' Tata' , wavelets_type)
	if wavelets_type == None:
		wavelets_type = 'db1'
	if evt_csv_name == None:
		evt_csv_name = '20051115T121100KAPCOR.csv'
	else:
		evt_csv_name = evt_csv_name[:-4] + '.csv'

	print('ds update wvlt : ', evt_csv_name)
	
	wavelets_graph = get_wavelets_and_acc_figures(evt_csv_name, member = 'MSG', wavelet_type = wavelets_type, wavelet_level = 7)
	wavelets_graph_2 = get_wavelets_and_acc_figures(evt_csv_name, member = 'MSD', wavelet_type = wavelets_type, wavelet_level = 7)	
	
	return wavelets_graph, wavelets_graph_2

'''
MAJ des graphs
'''
@app.callback(

	dash.dependencies.Output("ecg-graph", "figure"),
	dash.dependencies.Output("ecg-graph-2", "figure"),
	dash.dependencies.Output("graph5", "figure"),
	dash.dependencies.Output("graph6", "figure"),
	dash.dependencies.Input("patient-filter", "value"),
	dash.dependencies.Input("seizure-filter", "value"),
	dash.dependencies.Input("event-filter", "value"),
)
def update_graphs(patient, seizure, eventfile):

	mask = (
		((data.patient == patient) | (patient == None)) 
		& ((data.STY == seizure) | (seizure == None)) 
	)
	data_filtered = data.loc[mask, :]
	ecg_graph_figure = plotly.express.scatter(data_filtered,x="patient",y="STY",size='nb_seizure',color = 'SLO')

	event_test_fn = "20051115T121100KAPCOR.evt"

	try:
		csv_path = DIR_TAGGED_EVT + eventfile[:-4] + '.csv'	
		evt_o = EvtFile(eventfile)
		df_graf = pd.read_csv(csv_path,index_col = 0)
		ecg_graph_figure_2  = Func.add_figure_annotations(evt_o,df_graf)

	except:
		csv_path = DIR_TAGGED_EVT + '20051115T121100KAPCOR.csv'
		evt_o = EvtFile(event_test_fn)
		ecg_graph_figure_2 = empty_plot('')
	
	LIST_NORMES = ['MSG','MSD','T'] 
	columns_y_for_graph = ['seizure'] + LIST_NORMES

	csv_name = "20051115T121100KAPCOR.csv"
	# graph_figure_3 = get_wavelets_and_acc_figures(csv_name) 
        
    # graph3Content = plot_spectrogramme(evt_o.filename)
	# graph4Content = plot_spectrogramme(evt_o.filename, DIR_TAGGED_EVT, 'MSD')
	graph5Content = plot_psd(evt_o.filename)
	graph6Content = plot_psd(evt_o.filename, DIR_TAGGED_EVT, 'MSD')

	return ecg_graph_figure, ecg_graph_figure_2, graph5Content, graph6Content #, graph_figure_3 # graph4Content #, graph6Content 

###############
# Application #
###############

if __name__ == "__main__":
	app.run_server(port=8051, debug=True)

