import os
import sys
import numpy as np
import pandas as pd
from numpy.fft import fft


def reset_timeline(df):
    
    ts0 = df.timeline.iloc[0]
    df.timeline = [((i - ts0) / 1000) for i in df.timeline.tolist()] 

    return(df)

def get_epoch_df(df,no_epoch,duree_epoch):
    
    '''
    return a df of duree_epoch long centerd on 
    no_epoch * duree_epoch / 2
    '''
    window_overlap = duree_epoch / 2
    df_loc = df[(df.timeline > no_epoch * window_overlap) & 
    (df.timeline < (no_epoch+2) * window_overlap)]
    
    return(df_loc)

def make_model_df(df_data_csv,list_features,epoch_lenght): 
    nb_epoch = get_nb_epoch(df_data_csv,epoch_lenght)
    df_model = pd.DataFrame(data = np.zeros((nb_epoch,len(list_features))),columns = list_features)
    return(df_model,nb_epoch)

def get_time_segment_df(df,inter_time):
    df = df[(df.timeline >= inter_time[0]) & (df.timeline < inter_time[1])]
    return(df)

def make_model_df_from_csv(csv_name, dir_csv_model):

    '''
    make a csv for feature engineering from original datas
    center MSG & MSD data
    '''
    columns_variables_original = ['timeline','seizure','MSG','MSD']
    df_data_csv = pd.read_csv(dir_csv_model + csv_name,index_col = 0)
    df_data_csv = df_data_csv[columns_variables_original] 
    df_data_csv = df_data_csv.reset_index(drop=True)
    df_data_csv = reset_timeline(df_data_csv)

    mean_msg = df_data_csv['MSG'].mean()
    mean_msd = df_data_csv['MSD'].mean()

    df_data_csv['MSG'] = df_data_csv['MSG'] - df_data_csv['MSG'].mean()
    df_data_csv['MSD'] = df_data_csv['MSD'] - df_data_csv['MSD'].mean()

    return(df_data_csv,mean_msg,mean_msd)

def return_norm_L1(data):
    return(np.mean(np.absolute(data)))

def return_up_and_down(data):
    epsilon = 0.1
    # nb_min = (sum((i) > epsilon for i in list(data)) 
    # nb_max = (sum((i) < epsilon for i in list(data))
    # nb_ar = min(nb_min,nb_max)
    # return(nb_ar)
    return(sum(abs(i) > 0.1 for i in list(data)))
    
def get_dic_time_features_functions():
    
    dic_time_features_functions = dict()
    dic_time_features_functions['mean'] = np.mean
    dic_time_features_functions['std'] = np.std
    dic_time_features_functions['min'] = np.min
    dic_time_features_functions['max'] = np.max
    dic_time_features_functions['mean_absolute'] = return_norm_L1
    dic_time_features_functions['up_down'] = return_up_and_down

    return(dic_time_features_functions)

def get_nb_epoch(df, epoch_lenght):
    '''
    return the number of ecpoch from a timeline in second and a 
    givel overlap which is half the Epoch size
    2 must divide epoch_lenght
    '''
    window_overlap = epoch_lenght / 2
    timeline = df.timeline.tolist()
    duree_csv_sec = timeline[-1] - timeline[0]
    nb_epoch = int((duree_csv_sec // window_overlap) - 1)
    return(nb_epoch)

def get_times_features_name(list_members,list_time_features):

    list_members_times_features = []
    for member in list_members:
        for feature in list_time_features:
            new_feat = feature + '_' + member
            list_members_times_features.append(new_feat)
            
    return(list_members_times_features)

def get_entropy_features_name(list_members,list_entropy_features):

    list_members_entropy_features = []
    for member in list_members:
        for feature in list_entropy_features:
            new_feat = feature + '_' + member
            list_members_entropy_features.append(new_feat)
            
    return(list_members_entropy_features)

def make_all_features_list(list_time_features,list_fourier_features,list_entropy_features,meta_data = True):

    '''
    from :
    - list_time_features
    - list_fourier_features
    - list_entropy_features
    - list_evt_infos_columns 

    return :
    -  list of all model features names 
    '''
    if meta_data:
        list_evt_infos_columns = ['evt_name','epoch_nb','label']
    else:
        list_evt_infos_columns = []
        
    list_all_features = list_evt_infos_columns + list_time_features + list_fourier_features + list_entropy_features

    return(list_all_features)
 
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# -----------------------------------Time features-----------------------------------

def get_times_features_value(df_data,list_members,dic_tf_functions): 
    ''' 
    Dans dic y a une liste de nom et de fonctions
    df_features c'est le nouveau df
    df_data c'est les donnée sources
    '''
    dic_features_value = dict()
    for member in list_members:
        for key in dic_tf_functions.keys():
            str_feature = member + key
            dic_features_value[str_feature] = dic_tf_functions[key](df_data[member]) 
    features_value = list(dic_features_value.values())
    
    return(features_value)
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# -----------------------------------Entropy features-----------------------------------

def get_entropy_value(liste_acc,nb_entropy_range,member):
    '''
    liste_acc MSG or MSD
    nb_entropy_range is the number of range to calculate density probability 
    '''
    liste_acc = [i**2 for i in liste_acc]
    range_entropy = list(np.linspace(0,3,nb_entropy_range))
    nb_el = len(liste_acc)
    dispatch_list = pd.cut(liste_acc, range_entropy).value_counts()
    dispatch_list = dispatch_list / nb_el
    entropy = 0
    for freq in dispatch_list:
        if freq == 0:
            continue
        else:
            entropy = entropy + freq * np.log2(freq)

    return(entropy)

def get_entropy_from_epoch(df_epoch,nb_ent_range,member):
    
    '''
    return 3 entropy values from an epoch dataframe 
    input : 
    - the epoch df
    - the number of classe to calculate entropy, must be an integer

    output : 
    - entropy of middle left tab
    - entropy of middle right tab
    - entropy of total tab

    '''
    
    nb_el = len(df_epoch)
    mid_tab = int(round(nb_el/ 2,0))
    df_deb = df_epoch.iloc[0:mid_tab]
    df_end = df_epoch.iloc[mid_tab:]
    left_ent = get_entropy_value(df_deb,nb_ent_range,member)
    right_ent = get_entropy_value(df_end,nb_ent_range,member)
    glob_ent = get_entropy_value(df_epoch,nb_ent_range*2,member)

    return([glob_ent,left_ent,right_ent])

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

'''
Fourier features
'''

def get_fourier_features_names(list_intervalle_frequence,list_member,with_relatives = False):

    '''
    inpout : a list with intervalle like [0-3,3-7] with number in herz

    return 2 lists : 
    - one with fourier features name
    - one with relative fourier features name
    '''

    list_all_f_features  = []
    nb_features = len(list_intervalle_frequence)
    
    for member in list_member:
        list_fourier_features = []
        list_relative_fourier_features = []
        for i in range(nb_features - 1):
            low = list_intervalle_frequence[i]
            up = list_intervalle_frequence[i + 1]
            feature = member + '_' + 'F_' + str(low) + '-' + str(up)
            list_fourier_features.append(feature)
            if with_relatives:
                list_relative_fourier_features.append(feature + 'rel')
        list_all_f_features = list_all_f_features + list_fourier_features + list_relative_fourier_features

    return(list_all_f_features)

def get_df_spectre(df,col_name):

    '''
    - La freq_echantillonage c'est le nb de mesure qui decoupe la periode elementaire
    (typiquement 1 sec) dont je cherche un multiple, je définis par défaut ma période à partir
    de la longueur de la fréquence d'échantillonage, ici ma freq c'est 50ms et ma
    période élémentaire c'est 1 sec
    - La fonction retourne un df avec le spectre exprimé en hertz

    - Je récupère d'abord la vraie fréquence d'acquisition : freq_acquisition
    qui sera ma fréquence d'échantillonage
    '''

    nb_obs = len(df)
    t0 = df.iloc[0,0]
    tn = df.iloc[nb_obs-1,0]
    freq_echantillonage = (nb_obs / (tn-t0)) 
    # if abs(freq_echantillonage) < 44:
    #     print('attention freq_echantillonage = ', freq_echantillonage)

    x = df[col_name].to_numpy()
    N = len(x)

    tfd_x = fft(x)
    spectre_x = np.absolute(tfd_x)*2/N
    axe_frequences_elementaires = np.arange(N) / N * freq_echantillonage

    df_spectre = pd.DataFrame(columns=['frequence','valeur_coef'])
    df_spectre['frequence'] = axe_frequences_elementaires
    df_spectre['valeur_coef'] = spectre_x

    return(df_spectre)

def get_fourier_features_value(df_spectre,list_frequence):

    '''
    A partir d'un df avec les coef de fourier rattachés aux fréquences
    récupére les sommes de fourier pour construire les features
    '''

    list_values = []
    nb_features = len(list_frequence) - 1

    for i in range(nb_features):
        low = list_frequence[i]
        up = list_frequence[i + 1]
        df_loc = df_spectre[(df_spectre['frequence'] >low) & (df_spectre['frequence'] <= up)]
        somme_freq = df_loc['valeur_coef'].sum()
        list_values.append(somme_freq)

    '''
    on réupère aussi l'intégralité du spectre en le projetant sur 1-23hz en découpant
    par tranche de 1hz
    '''
    full_spectre_1hz = []

    for i in np.arange(12)*2:
        low = i
        up = i + 1
        df_loc = df_spectre[(df_spectre['frequence'] >low) & (df_spectre['frequence'] <= up)]
        somme_freq = df_loc['valeur_coef'].sum()
        full_spectre_1hz.append(somme_freq)

    return(list_values,full_spectre_1hz)

# def get_fourier_features_relative_value(df_spectre,list_frequence):

#     '''
#     A partir d'un df avec les coef de fourier rattachés aux fréquences
#     récupére les sommes de fourier pour construire les features
#     '''

#     list_values = []
#     list_relative_values = []
#     nb_features = len(list_frequence)
#     somme_coef = np.sum(df_spectre['valeur_coef'])

#     for i in range(nb_features - 1):
#         low = list_frequence[i]
#         up = list_frequence[i + 1]
#         df_loc = df_spectre[(df_spectre['frequence'] >low) & (df_spectre['frequence'] <= up)]
#         somme_freq = df_loc['valeur_coef'].sum()
#         list_values.append(somme_freq)
#         list_relative_values.append(somme_freq / somme_coef)

#     return(list_relative_values)

def get_epoch_fourier_features(df_epoch,member,list_intervalle_frequence_fourier):
    '''
    return list of fourier features values from 
    a list of intervalle in hz list_intervalle_frequence_fourier
    '''
    df_spectre = get_df_spectre(df_epoch,member)
    list_features_value,full_spectre_1hz = get_fourier_features_value(df_spectre,list_intervalle_frequence_fourier)

    return(list_features_value,full_spectre_1hz)  

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# WAVELET 

def return_frequency_dataframe(global_wv_signal):
    '''
    requiere un signal calibréà 46hz
    '''
    nb_scales = len(global_wv_signal) # il y autant de scaling que de lignes
    nb_cols_total = len(global_wv_signal[-1]) * 2 # le nombre de col par défaut c'est celui de la plus haute fréquence

    df_columns = list(range(nb_cols_total))
    list_frequency = ['46-23','23-11.5','11.5-5.75','5.75-2.8','2.8-1.4','1.4,0.7','<0.7']
    df_wavelet = pd.DataFrame(index = list_frequency , columns = df_columns) # je numerote mes colonnes pour les afficher ds la heatmap 
    ind_freq = 0 # index = list_frequency[0:nb_scales],

    for i,freq_row in enumerate(np.arange(nb_scales -1,0,-1)):
        
        band_wavelet_signal = global_wv_signal[freq_row]
        band_wavelet_signal = band_wavelet_signal / (np.sqrt(2)**(i+1))
        
        # band_wavelet_signal =  duplicate_list(band_wavelet_signal,1)  

        frequency = list_frequency[ind_freq]
                    
        if freq_row == 1: # special treatment for the last 2 rows
            nb_col = 2**(i+1)
            band_wavelet_signal =  duplicate_list(band_wavelet_signal,nb_col)  
            df_wavelet.loc[frequency] = band_wavelet_signal[0:nb_cols_total] 
        
        else:
            nb_col = 2**(i+1)
            band_wavelet_signal =  duplicate_list(band_wavelet_signal,nb_col)  
            band_wavelet_signal = band_wavelet_signal[0:nb_cols_total] 
            df_wavelet.loc[frequency] = band_wavelet_signal
        
        ind_freq += 1

        df_wavelet = df_wavelet.astype(float)
        df_wavelet = df_wavelet.reindex(index=df_wavelet.index[::-1])

    return df_wavelet

# def get_wavelets_and_acc_figures(csv_name, member = 'MSD', wavelet_type = 'db1', wavelet_level = 7):
#     '''
#     return a df for graf and e discret wavelet heatmap, arguments are : 
#         - acc data csv name
#         - member (MSG or MSD)
#         - discret wavelet trype
#         - number of coefficient level required
#     '''
#     csv_path = DIR_TAGGED_EVT + csv_name
#     df_graf = pd.read_csv(csv_path,index_col = 0)
#     df_graf.timeline = [round(i) - df_graf.timeline.iloc[0] for i in list(df_graf.timeline)]

#     np_x = df_graf[member].to_numpy()
#     np_x = np_x - np.mean(np_x) #pour ne pas charger les plus basses fréquences
#     # np_x = np_x[10000:15000]
#     print(wavelet_type)
#     wv_signal = pywt.wavedec(np_x,wavelet_type,level = wavelet_level)
#     df_frequency_for_heatmap = return_frequency_dataframe(wv_signal)
#     frequency_band = list(df_frequency_for_heatmap.index)