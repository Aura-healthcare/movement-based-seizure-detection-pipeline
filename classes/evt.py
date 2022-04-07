import sys
import pandas as pd
import numpy as np

sys.path.insert(0, '..')
from config import *
from utils import utils

# Construction of a class to get easy informations and manipulate evt files

class EvtFile:
    def get_df_event_list(self, evt_adress):
    
        df_event = pd.DataFrame(columns = ['STI','ETI','STY','SLO'])
        no_row = 0
        
        with open(evt_adress , "r",encoding="ISO-8859-1") as f: 
            for i,line in enumerate(f):
                if i > 2:
                    
                    ligne_event = line.split(";")
                    STI = utils.evt_data_to_ms(ligne_event[1])
                    ETI = utils.evt_data_to_ms(ligne_event[2])
                    STY = ligne_event[4].strip()
                    SLO = ligne_event[6].strip()
                    df_event.loc[no_row] = [STI,ETI,STY,SLO]
                    no_row += 1

        df_event = df_event.sort_values(by = 'STI', ascending = True)
        df_event = df_event.reset_index()
        df_event = df_event.drop('index',axis=1)

        return(df_event)

    def __init__(self, filename, dir_event_file = DIR_EVT_FILES):
        
        self.filename = filename
        self.adress = dir_event_file + filename
        self.edfname = ''
                
        if len(self.filename) == 25:
            self.patient = self.filename[15:18]
        else:
            self.patient = self.filename[15:18]

        df_event = self.get_df_event_list(self.adress)
        list_ts = df_event.STY.tolist()
        
        try:
            ind_t0 = list_ts.index('t0')
            ind_t1 = list_ts.index('t1')
            ind_t2 = list_ts.index('t2')
            if (ind_t0 < ind_t1) and (ind_t1 < ind_t2):
                self.ts_order = 'True'
            else:
                self.ts_order = 'False'
        except: 
            self.ts_order = 'Missing_ts'

        df_event = self.get_df_event_list(self.adress)
        self.EVT_start = np.min(df_event['STI'])
        self.EVT_end = np.max(df_event['ETI'])

        self.STI = df_event['STI'].tolist()
        self.ETI = df_event['ETI'].tolist()
        self.STY = df_event['STY'].tolist()
        self.SLO = df_event['SLO'].tolist()
        self.EVT_Duration = int(self.EVT_start) - int(self.EVT_end)
        self.nb_evts = len(self.STI)  

        try:
            match_evt_edf = pd.read_csv(MATCH_EVT_EDF_RELATIVE_PATH, index_col=0)
            dic_evt_edf = utils.get_evt_to_edf_dict(match_evt_edf)
            self.edfname = dic_evt_edf[self.filename ][:-4]
        except:
            toto = 1 

