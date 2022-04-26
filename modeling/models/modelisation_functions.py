from sklearn.metrics import confusion_matrix
from sklearn import metrics 
import pandas as pd
import matplotlib.pyplot as plt 

def discretize_label(df,treshold):
    '''
    label 1,2 -> 1
    '''
    df['label_bin'] = [int(i>treshold) for i in df.label.tolist()]
    
    return(df)

def get_x_test_y_test(df,features):
    '''
    return x et y from dataframe
    '''
    x_test = df[features]
    y_test = df['label_bin']
    
    return(x_test,y_test)

def loso(df_data,sz_name,list_features_model):
    '''
    leave one seizure out
    return train_test df without and with seizure
    '''
    df_train = df_data[df_data.evt_name != sz_name]
    df_test = df_data[df_data.evt_name == sz_name]

    return(df_train[list_features_model],df_test[list_features_model],df_train['label_bin'],df_test['label_bin'])

def return_sens_and_spec(y_test,y_pred):
    
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    sensibilité = tp / (tp + fn)
    spécificité = tn / (tn + fp)

    return(sensibilité,spécificité)

def plot_roc_curve(y_test,y_pred,graph_curve = True):
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
    if graph_curve:
        plt.figure(figsize=(9,6))
        plt.plot(fpr,tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('ROC CURVE')
        plt.show()
    return(fpr,tpr)

def proba_to_bin(y_pred_proba,treshold):
    '''
    input :
    - list proba
    - treshold
    '''
    y_pred_bin = [int(i>treshold) for i in y_pred_proba]
    return(y_pred_bin)

def df_from_seizure_type(list_seizure,data,bdd):
    '''
    return a dataframe with only seizure of selected type
    input : 
    - list of seizure type
    output : 
    - the dataframe
    - the list of sz
    '''
    list_sz = bdd[bdd.STY.isin(list_seizure)].filename.unique()
    list_sz = [i[:-4] for i in list_sz]
    data = data[data.evt_name.isin(list_sz)]
    list_sz = data.evt_name.unique()

    return(data,list_sz)