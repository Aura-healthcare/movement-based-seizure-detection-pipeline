
import sys 
import pyedflib 

sys.path.insert(0, '..')
from config import *

# Construction of a class to get easy informations and manipulate edf files

class EdfFile:
    def __init__(self, filename, dir_edf_file = DIR_EDF_FILES):

        year_edf = filename[0:4]   
        self.adress = dir_edf_file + year_edf + '/' + filename
        self.patient = filename[15:18]
        self.date = filename[0:8]

        try:
            f = pyedflib.EdfReader(self.adress)
        except:
            f._close()
            f = pyedflib.EdfReader(self.adress)   
    
        start_datetime = f.getStartdatetime()
        self.start_datetime_ms = int(start_datetime.timestamp() * 1000)  
        self.exam_duration_ms = f.getFileDuration() * 1000
        self.end_datetime_ms = self.start_datetime_ms +  self.exam_duration_ms

        try:
            f._close()
            del f
        except:
            pass