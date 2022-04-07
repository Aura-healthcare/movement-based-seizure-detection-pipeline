import sys
import os
from datetime import datetime 

sys.path.insert(0, '..')
from config import * 

def get_evt_to_edf_dict(EvtBDG):
    dict_evt_to_edf = dict(zip(EvtBDG.evt_filename, EvtBDG.edf_filename))
    return(dict_evt_to_edf)

def get_evt_filename_list(dir_evt = DIR_EVT_FILES):
    return((os.listdir(DIR_EVT_FILES)))

def get_edf_filename_list(dir_edf = DIR_EDF_FILES, list_years_edf = LIST_YEARS_EDF):
    list_edf = []
    for year in list_years_edf:
        edf_dir = dir_edf + year +'/'
        for f in os.listdir(edf_dir):
            if str(f[-3:]) =='edf':
                list_edf.append(f)
    
    return(list_edf)

#Pour récupérer la date des formats .evt

def evt_data_to_ms(evt_ts):
    evt_ts = datetime.strptime(evt_ts.strip(), '%Y%m%dT%H%M%S,%f')
    millisec = evt_ts.timestamp() * 1000
    return(millisec)