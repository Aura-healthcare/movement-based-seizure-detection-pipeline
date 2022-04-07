import numpy as np

def add_euclidean_norms(data, list_noms = ['MSG', 'MSD', 'T']):

    acc_x = ['C1ax+C1ax-', 'C1ay+C1ay-', 'C1az+C1az-']
    acc_y = ['C2ax+C2ax-', 'C2ay+C2ay-','C2az+C2az-']
    acc_z = ['C2az+C2az-','C3ax+C3ax-', 'C3ay+C3ay-', 'C3az+C3az-']
    
    list_3_emplacement = [acc_x,acc_y,acc_z]

    for list_3_axes,nom in zip(list_3_emplacement, list_noms): 
        data[nom] = np.sqrt(data[list_3_axes[0]]**2 + data[list_3_axes[1]]**2 + data[list_3_axes[2]]**2)

    return(data)

def get_sz_dataframe(df_edf_data, df_evt):

    index_crise = 0
    df_edf_data['seizure'] = index_crise
    
    timeline = df_edf_data.timeline

    for i,event in enumerate(df_evt.STY.tolist()):
        str_event  = str(event).strip()
        if str_event in ['t0','t1','t2']:
            ts_event = df_evt['STI'].iloc[i]
            index_crise +=1
            index_crise = index_crise % 3
            df_edf_data.loc[df_edf_data['timeline'] > ts_event,'seizure'] = index_crise

    return(df_edf_data) 