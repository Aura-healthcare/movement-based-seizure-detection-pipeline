import sys
import pandas as pd

sys.path.insert(0, '../..')
from config import *
from classes.evt import *
from utils import utils

# Parsing : Save file if specified in command line

try:
    if sys.argv[1] == 'True':
        save_file = True
    else:
        save_file = False
except:
    save_file = False

# Create the evt_BDG csv file and save it in '../Datas/BDG_exploitation/evt_BDG.csv'
    
# Input : 
#    - match_evt_edf : string
#    - save_file : boolean
    
# Output : 
#    - Save a csv 'match_evt_edf.csv in '../data/preprocessing/'  

def make_evtBDG_dataframe(match_evt_edf_filename, save_file):

    columns_BDG = ['filename', 'patient', 'STI', 'ETI', 'STY', 'SLO', 'edf_match_filename']
    df_EvtBDG = pd.DataFrame(columns = columns_BDG) 
    evt_to_edf = pd.read_csv(match_evt_edf_filename, index_col = 0)
    dict_evt_to_edf = utils.get_evt_to_edf_dict(evt_to_edf)
 
# Loop on every event filename to retrieve every event caracteristic and save it in EvtBDG.csv, 
# the match between edf and evt must have been prealably made

    row = 0
    list_evt_filenames = utils.get_evt_filename_list() 
    list_ok_evt = []

    for evt_filename in list_evt_filenames:
        
        evt_object = EvtFile(evt_filename)
        patient = evt_object.patient 
        edf_correspondant = dict_evt_to_edf[evt_filename]

        for no_event,event_type in enumerate(evt_object.STY):
            
            STI = evt_object.STI[no_event]
            ETI = evt_object.ETI[no_event]
            STY = evt_object.STY[no_event]
            SLO = evt_object.SLO[no_event]
            df_EvtBDG.loc[row] = [evt_filename,patient, STI, ETI, STY, SLO, edf_correspondant]
            row += 1

    if save_file :
        df_EvtBDG.to_csv(DIR_SAVE_PREPROCESSING_DATA + 'simplified_evt.csv')
        print('File simplified_evt.csv saved succesfully')

    else:
        print('The code as run ok but no file have been saved, True must be passed as argument to save file ')

if __name__ == "__main__":

    match_evt_edf_filename = MATCH_EVT_EDF_PATH
    make_evtBDG_dataframe(match_evt_edf_filename, save_file)