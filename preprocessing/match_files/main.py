import sys
import pandas as pd

sys.path.insert(0, '../..')
from config import *
from classes.edf import *
from classes.evt import *
from utils import utils

# Parsing : Save file if specified in command line

try:
    if sys.argv[1] == 'True':
        save_file = True
        print('Filesave ok')
    else :
        save_file = False
except:
    save_file = False
    print('There will be no filesave')

 
# Creation of a dataframe with edf files and their patient + timeline
# then we can check if one event for one patient is inside the timeline 

# Input : 
#    - No input

# Output :
#    - Dataframe with infos on edf

def get_edf_dataframe_with_infos():

    print('Creation of a dataframe with edf files and their patient + timeline')
    print('This operation may take 20mn')

    list_edf_filenames = utils.get_edf_filename_list()

    df_edf = pd.DataFrame(columns = ['edf_filename', 'patient', 'start', 'end'])
    print('Number of edf files to loop on :', len(list_edf_filenames))
    
    for i, edf_filename in enumerate(list_edf_filenames):
        edf = EdfFile(edf_filename)
        patient = edf.patient
        start = edf.start_datetime_ms
        end = edf.end_datetime_ms
        df_edf.loc[i] = [edf_filename, patient, start, end]
        if i % 5 ==0:
            print(i)

    return(df_edf)

# Match evt with edf on patient Id plus timeline adequation
# i.e. events ts are inside edf timeline. 
    
# Input : 
#    - df_edf, a dataframe with edf files and their patient + timeline
#    - Boolean savefile to save the dataframe in a csv

# Output :
#    - Save a csv 'match_evt_edf.csv with matched evt and edf in '../Datas/BDG_exploitation/'  

def save_match_evt_with_edf(df_edf, filesave):
    
    df_match_evt_edf = pd.DataFrame(columns = ['evt_filename', 'edf_filename', 'nb_match'])
    list_no_match = []

    list_evt_filenames = utils.get_evt_filename_list()

    for i,evt_filename in enumerate(list_evt_filenames):
        
        evt_object = EvtFile(evt_filename)
        evt_patient = evt_object.patient
        evt_start = evt_object.EVT_start
        evt_end = evt_object.EVT_end

        df_match = df_edf[(df_edf.start < evt_start) & (df_edf.end > evt_end) 
        & (df_edf.patient == evt_patient)]
        
        if len(df_match) == 1:
            edf_filename = df_match.edf_filename.tolist()[0]
            edf_filename = df_match.edf_filename.tolist()[0]
            df_match_evt_edf.loc[i] = [evt_filename,edf_filename,1]

        if len(df_match) > 1:
            print('Error with {}, there are more than one match'.format(evt_filename))

        if len(df_match) == 0:
            df_match_evt_edf.loc[i] = [evt_filename, 'no_edf', 0]
            list_no_match.append(evt_filename)

    if  save_file:
        df_match_evt_edf.to_csv(DIR_SAVE_PREPROCESSING_DATA + 'match_evt_edf.csv')
        print('File match_evt_edf saved succesfully')

    print('Number correct of match = ', len(df_match_evt_edf[df_match_evt_edf.nb_match==1]))
    print('Number of no match = ', len(df_match_evt_edf[df_match_evt_edf.nb_match==0]))
    print('No edf file found for the followings evt files :')
    print(list_no_match)

if __name__ == "__main__":

    df_edf = get_edf_dataframe_with_infos()
    print('Download of edf files done, now matching edf file with evt file')
    save_match_evt_with_edf(df_edf, save_file)