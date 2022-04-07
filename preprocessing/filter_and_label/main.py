import sys
import os
import pyedflib 
import pandas as pd
import numpy as np
from compute import *

sys.path.insert(0, '../..')
from config import *
from classes.edf import *
from classes.evt import *
from utils import utils

# Return files items needed for labeling the edf

# Input : 
#    - Directory in which are saved datas

# Output :
#    - Create directory if necessary
#    - List of good evt that match an edf  

def get_list_evt_and_edf_to_label(dir_processed_data):

    DIR_SAVE_CSV_EDF =  dir_processed_data + 'filtered_and_labeled_edf/'
    DIR_SAVE_CSV_EVT = dir_processed_data + 'filtered_and_labeled_evt/'
    DIR_SAVE_CSV__STATS = dir_processed_data + 'seizure_statistics/'

    if not os.path.exists(DIR_SAVE_CSV_EDF):
        os.makedirs(DIR_SAVE_CSV_EDF)
        
    if not os.path.exists(DIR_SAVE_CSV_EVT):
        os.makedirs(DIR_SAVE_CSV_EVT)

    if not os.path.exists(DIR_SAVE_CSV__STATS):
        os.makedirs(DIR_SAVE_CSV__STATS)

    EvtBDG = pd.read_csv(SIMPLIFIED_EVT_PATH, index_col = 0)

    df_match_evt_edf = pd.read_csv(MATCH_EVT_EDF_PATH, index_col = 0)
    list_edf_to_tag = df_match_evt_edf.edf_filename.unique()

# List evt to label

    list_evt_to_tag = []

    for evt_filename in list(EvtBDG.filename.unique()):
        evt_o = EvtFile(evt_filename)
        ok = evt_o.ts_order
        if ok == 'True':
            list_evt_to_tag.append(evt_filename)

    return(list_edf_to_tag, list_evt_to_tag)

# Creating a dataframe
# df_edf_to_tag  with an evt correspondance list for each edf.  

# Input : 
#    - list_edf_to_tag and list_evt_to_tag as returned by get_list_evt_and_edf_to_label

# Output :
#    - A dataframe df_edf_to_tag with a list of related evt for each edf 

def create_df_with_edf_evt(list_edf_to_tag, list_evt_to_tag):

    df_match_evt_edf = pd.read_csv(MATCH_EVT_EDF_PATH, index_col = 0)
    df_edf_to_tag = pd.DataFrame(columns = ['edf_filename', 'list_evt'])
    row = 0

    for edf_filename in list(list_edf_to_tag):
        df_loc = df_match_evt_edf[df_match_evt_edf.edf_filename == edf_filename]
        list_evt = df_loc.evt_filename.tolist()
        list_edf_evt = list(set(list_evt) & set(list_evt_to_tag))
        
        if len(list_edf_evt) > 0:
            df_edf_to_tag.loc[row] = [edf_filename,list_edf_evt]
            row += 1
            
    df_edf_to_tag = df_edf_to_tag[df_edf_to_tag.edf_filename != 'no_edf']

    return(df_edf_to_tag)

# Get a dataframe with acceleration data centered on the seizure. 

# Input : 
#    - dataframe with acceleration data 
#    - required graph window size 
#    - the timestamp aof the neginning of the seizure 

# Output :
#    - Dataframe 

def loc_get_df_sz_graph(df_data,ts_begin_sz, graph_window_size = 600000):
# To be sure we have the good graph_window_size we must chexk the file is long enough before the seizure begin

    ts_min = df_data.loc[df_data.index[0],'timeline']
    ts_begin_graph = np.max((ts_min, ts_begin_sz - graph_window_size / 2))
    ts_end_graph = ts_begin_graph + graph_window_size
    
    loc_df_graph = df_data[(df_data['timeline'] > ts_begin_graph) 
    & (df_data['timeline'] < ts_end_graph)]
    
    return(loc_df_graph)

# Save all labeled edf in csv in DIR_SAVE_CSV_EDF and DIR_SAVE_CSV_EVT

# Input : 
#    - df_edf_to_tag from create_df_with_edf_evt 

# Output :
#    - Save labeled csv files in DIR_SAVE_CSV_EDF and DIR_SAVE_CSV_EVT
#    - Save statistics in DIR_SAVE_CSV__STATS + 'edf_acceleration_statistics.csv'  

def save_labeled_csv(df_edf_to_tag, save_file, graph_window_size):

    it_evt, row_stat = 0, 0
    columns_stats = ['evt_name', 'edf_name', 'mean_MSG', 'mean_MSD', 'mean_T', 'seizure_duration', 'std_G', 'std_D', 'std_T']
    df_stats = pd.DataFrame(columns=columns_stats)

    DIR_SAVE_CSV_EVT = DIR_TAGGED_EVT
    DIR_SAVE_CSV_EDF = DIR_TAGGED_EDF

    for row,edf_filename in enumerate(df_edf_to_tag.edf_filename.tolist()):
        
        list_evt = df_edf_to_tag.iloc[row,1]
        edf_o = EdfFile(edf_filename)
        adr_edf_to_tag = edf_o.adress

        df_glob_evt = pd.DataFrame(columns = ['STI', 'ETI', 'STY', 'SLO', 'no_evt'])

        for no_evt,evt_fn in enumerate(list_evt): 
            evt_o = EvtFile(evt_fn)
            df_loc_evt = evt_o.get_df_event_list(evt_o.adress)
            df_loc_evt['no_evt'] = no_evt
            df_glob_evt = pd.concat([df_glob_evt,df_loc_evt], ignore_index=True)
        
        df_glob_evt = df_glob_evt.sort_values('STI')
        df_glob_evt = df_glob_evt.reset_index()
        ts_begin_sz = df_glob_evt.loc[0, 'STI']
        ts_end_sz = df_glob_evt.loc[len(df_glob_evt) - 1, 'ETI']

#  Filter edf file because they've been artificially forced 256hz or 512hz

        EDF = EdfFile(edf_filename)
        duration_ms = EDF.exam_duration_ms
        start_datetime_ms = EDF.start_datetime_ms
        end_datetime_ms = start_datetime_ms + duration_ms

        adr_edf = EDF.adress
        f = pyedflib.EdfReader(adr_edf)
        signal_labels = f.getSignalLabels()

        df_acc_data = pd.DataFrame(columns = ACCELERO_LABELS)

        for label in ACCELERO_LABELS: 
            index_signal = signal_labels.index(label)
            signal = f.readSignal(index_signal)
            df_acc_data[label] = signal

        columns = list(df_acc_data.columns)
        
# Creating a timeline

        nb_obs = len(df_acc_data)
        timeline = np.linspace(start_datetime_ms, end_datetime_ms, nb_obs)
        df_acc_data['timeline'] = timeline

# Droping duplicate

        mask_drop_duplicate = df_acc_data[columns].duplicated()
        mask_drop_duplicate = [bool((True + bool(i)) % 2)  for i in mask_drop_duplicate]

        df_acc_data = df_acc_data.loc[mask_drop_duplicate, :]
        df_acc_data = add_euclidean_norms(df_acc_data)

        df_sz_edf = get_sz_dataframe(df_acc_data,df_glob_evt)


# Because some edf are constant for a long time in the begginning and the drop duplicate method have been used,
# the fiste instant must be suppressed to avoid constant acc data in the file
        
        df_sz_edf = df_sz_edf[1:]

        if row % 5 ==0:
            print('edf processed : ',row)
        
        if save_file:
            df_sz_edf.to_csv(DIR_SAVE_CSV_EDF + edf_filename[:-4] + '.csv')

            for no_evt,evt_fn in enumerate(list_evt): 
                
                it_evt += 1
                toto = df_glob_evt[df_glob_evt.no_evt == no_evt]
                ts_begin_sz = int(toto[toto.STY  =='t0']['STI'])
                ts_end_sz = int(toto[toto.STY  =='t2']['ETI'])

                means = df_sz_edf[['MSG','MSD','T']].mean()
                std = df_sz_edf[['MSG','MSD','T']].std()
                df_stats.loc[row_stat] =  [evt_fn,edf_filename] + list(means) + [(int(ts_end_sz) - int(ts_begin_sz)) / 1000] + list(std)
                row_stat += 1 

                df_sz_evt = loc_get_df_sz_graph(df_sz_edf,ts_begin_sz, graph_window_size)
                df_sz_evt.to_csv(DIR_SAVE_CSV_EVT + evt_fn[:-4] + '.csv')

if __name__ == "__main__":

    try:
        save_file = sys.argv[1] == 'True'
    except:
        save_file = False

    try:
        if int(sys.argv[2]) > 0:
            graph_window_size = int(sys.argv[2])
            print('gws = ',graph_window_size)
        else:
            print('Wrong window size, must be an integer, defaut = 600 seconds')
            graph_window_size = 600000
    except:
        print('No graph window size specified, defaut = 600 seconds')
        graph_window_size = 600000

    dir_processed_data = DIR_SAVE_PROCESSED_DATA

    list_edf_to_tag, list_evt_to_tag = get_list_evt_and_edf_to_label(dir_processed_data)
    df_edf_to_tag = create_df_with_edf_evt(list_edf_to_tag, list_evt_to_tag)
    save_labeled_csv(df_edf_to_tag,save_file,graph_window_size)
    
    if save_file:
        print('All files saved succesfully')
    else:
        print('All edf files processed succesfully but not saved')