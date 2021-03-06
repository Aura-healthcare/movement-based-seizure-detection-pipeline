import base64
import pylab
import pywt
import sys
import io
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt 
from spectrum import Spectrogram
from spectrum import Periodogram

import sys
sys.path.insert(0, '../..')
from utils import utils
from config import *

# Return psd from evt filename, if member not specified then 'MSG' 

def plot_psd(evt_name, DIR_TAGGED_EVT = '../../data/filtered_and_labeled_evt/', member = 'MSG'):

    csv_path = DIR_TAGGED_EVT + evt_name[:-4] + '.csv'
    df_graf = pd.read_csv(csv_path, index_col = 0)

    np_x = df_graf[df_graf.seizure>0][member].to_numpy()
    np_x = np_x - np.mean(np_x)

    p = Periodogram(np_x, sampling=46)
    df_psd = pd.DataFrame(data= np.transpose([p.frequencies(), p.psd]),columns=['Fréquence en hz','Puissance'])
    figure = px.line(df_psd, x="Fréquence en hz",y="Puissance",title="Power spectral density within seizure frametime")

    return(figure)

# Return a spectrum
# Argument : 
#    - Complete evt filename like toto.evt
#    - Member : 'MSG' or 'MSD'
#    - Sampling = 10 size of each sub-window 
#    - ws Size of the window on wich i compute fourier
#    - W c'est la résolution i.e. le nb de points dont il se sert pour Fourier, la plus grand W la plus fine la résolution

def plot_spectrogramme(evt_filename, DIR_TAGGED_EVT = '../../data/filtered_and_labeled_evt/', member = 'MSG',
    sampling = 46, window_size = 5, granularité = 512, ylim = [0,15]):

    csv_path = DIR_TAGGED_EVT + evt_filename[:-4] + '.csv'	
    df_graf = pd.read_csv(csv_path,index_col = 0)
    np_x = df_graf[df_graf.seizure>0][member].to_numpy()
    np_x = np_x - np.mean(np_x)

    p = Spectrogram(np_x, ws=window_size, W=granularité, sampling = sampling) 
    fig = plt.figure(figsize = (10, 7))
    p.periodogram() 
    p.plot()
    pylab.ylim(ylim)
    pylab.title("Crise {}, membre {} \n Fréquence en ordonnées de {} à {} hz".format(evt_filename[:-4], member, ylim[0], ylim[1]))

    buf = io.BytesIO() # in-memory files
    plt.tight_layout()
    plt.savefig(buf, format = "png") # save to the above file object
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    return "data:image/png;base64,{}".format(data)


def duplicate_list(liste,nb_col):
    
    liste = list(liste)
    new_liste = []
    for el in liste:
        loc = [el] * nb_col
        new_liste = new_liste + loc
    
    new_liste = [float(np.abs(i)) for i in new_liste]

    return(new_liste)

# Require a 46Hz signal

def return_frequency_dataframe(global_wv_signal):

    nb_scales = len(global_wv_signal) # Same number of scaling and lines
    nb_cols_total = len(global_wv_signal[-1]) * 2 # Default number of column is the one of the highest frequencies

    df_columns = list(range(nb_cols_total))
    list_frequency = ['46-23', '23-11.5', '11.5-5.75', '5.75-2.8', '2.8-1.4', '1.4,0.7', '<0.7']
    df_wavelet = pd.DataFrame(index = list_frequency, columns = df_columns) # Numbering columns for the heatmap 
    ind_freq = 0

    for i,freq_row in enumerate(np.arange(nb_scales -1, 0, -1)):
        
        band_wavelet_signal = global_wv_signal[freq_row]
        band_wavelet_signal = band_wavelet_signal / (np.sqrt(2)**(i+1))
        
        frequency = list_frequency[ind_freq]
                    
        if freq_row == 1: # Special treatment for the last 2 rows
            nb_col = 2**(i+1)
            band_wavelet_signal =  duplicate_list(band_wavelet_signal, nb_col)  
            df_wavelet.loc[frequency] = band_wavelet_signal[0:nb_cols_total] 
        
        else:
            nb_col = 2**(i+1)
            band_wavelet_signal =  duplicate_list(band_wavelet_signal, nb_col)  
            band_wavelet_signal = band_wavelet_signal[0:nb_cols_total] 
            df_wavelet.loc[frequency] = band_wavelet_signal
        
        ind_freq += 1

        df_wavelet = df_wavelet.astype(float)
        df_wavelet = df_wavelet.reindex(index=df_wavelet.index[::-1])

    return df_wavelet

# Return a dataframe for graf and a discret wavelet heatmap, arguments are : 
#    - acc data csv name
#    - member (MSG or MSD)
#    - discret wavelet trype
#    - number of coefficient level required

def get_wavelets_and_acc_figures(csv_name, member = 'MSD', wavelet_type = 'db1', wavelet_level = 7):

    csv_path = DIR_TAGGED_EVT + csv_name
    df_graf = pd.read_csv(csv_path,index_col = 0)
    df_graf.timeline = [round(i) - df_graf.timeline.iloc[0] for i in list(df_graf.timeline)]

    np_x = df_graf[member].to_numpy()
    np_x = np_x - np.mean(np_x) # To not overcharge low frequencies
    wv_signal = pywt.wavedec(np_x,wavelet_type,level = wavelet_level)
    df_frequency_for_heatmap = return_frequency_dataframe(wv_signal)
    frequency_band = list(df_frequency_for_heatmap.index)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("", "<b> wavelet type = {} </b>".format(wavelet_type))
    )
    
    fig.add_trace(go.Scatter(x = df_graf.timeline, y = df_graf[member]), 1,1)
    fig.add_trace(go.Scatter(x = df_graf.timeline, y = df_graf.seizure), 1,1)
    fig.add_trace(go.Heatmap(x = df_graf.timeline, z = np.asarray(df_frequency_for_heatmap),
                        y=frequency_band,
                        hoverongaps = False,
                        coloraxis = "coloraxis",showlegend=True), 2,1)
    fig.update_yaxes(title_text= 'Accélération, 1 = G', row=1, col=1)
    fig.update_yaxes(title_text= 'Bande de fréquences en hz', row=2, col=1)
    fig.update_layout(autosize = True, title_text="<b> {}, data and wavelets </b>".format(member,wavelet_type))
    fig.update_layout(autosize=True, height = 1000, showlegend=False)

    return(fig)

# Plot Continous Wavelet

def return_continous_wavelet_fig(csv_name, member = 'MSD', wavelet_type = 'gaus1'):

    csv_path = DIR_TAGGED_EVT + csv_name
    df_graf = pd.read_csv(csv_path, index_col = 0)
    df_graf.timeline = [round(i) - df_graf.timeline.iloc[0] for i in list(df_graf.timeline)]
    np_x = df_graf[df_graf.seizure>0][member].to_numpy()
    np_x = np_x - np.mean(np_x)
 
    N = len(np_x)
    dt = 1 / 46
    time = np.arange(0, N) * dt

    scales = np.arange(1,256)
    figure = figure_wavelet(time, np_x, scales, sz_name = csv_name)

    return(figure)

def figure_wavelet(time, signal, scales,sz_name, waveletname = 'gaus1', 
                cmap = plt.cm.seismic, 
                ylabel = 'Période en secondes : freq = 1 / période', 
                xlabel = 'Time'):

        title = 'Transformée en ondelettes {} du signal de {}'.format(waveletname, sz_name), 
        dt = time[1] - time[0]
        [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
        power = (abs(coefficients)) ** 2
        period = 1. / frequencies
        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
        contourlevels = np.log2(levels)
        
        fig, ax = plt.subplots(figsize=(15, 10))
        im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both', cmap=cmap)
        
        ax.set_title(title, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_xlabel(xlabel, fontsize=18)
        
        yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels(yticks)
        ax.invert_yaxis()
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], -1)
        
        cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
        fig.colorbar(im, cax=cbar_ax, orientation="vertical")
        
        return(fig)

# Returns an empty plot with a centered text.

def empty_plot(label_annotation):
	trace1 = go.Scatter(
		x=[],
		y=[]
	)

	data = [trace1]

	layout = go.Layout(
		showlegend=False,
		xaxis=dict(
			autorange=True,
			showgrid=False,
			zeroline=False,
			showline=False,
			ticks='',
			showticklabels=False
		),
		yaxis=dict(
			autorange=True,
			showgrid=False,
			zeroline=False,
			showline=False,
			ticks='',
			showticklabels=False
		),
		annotations=[
			dict(
				x=0,
				y=0,
				xref='x',
				yref='y',
				text=label_annotation,
				showarrow=True,
				arrowhead=7,
				ax=0,
				ay=0
			)
		]
	)

	fig = go.Figure(data=data, layout=layout)
	
	return fig