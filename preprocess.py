import pandas as pd
import numpy as np
import re
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from module import dataset

# Integration Total dataset

data_path = '/VOLUME/nia_vent_weaning/data/'



def main(data_path, input_window):

    '''
        Return : (i) Labelling DataFrame  (ii) dataset DataFrame
    '''

    label = dataset.get_label(data_path + 'integrated_data/before/')
    data_dir = data_path + '/model_data/' + str(input_window) + 'h/'
    df1 = pd.read_csv(data_dir + '0h_data.csv', index_col=0)
    df2 = pd.read_csv(data_dir + '1h_data.csv', index_col=0)
    df3 = pd.read_csv(data_dir + '2h_data.csv', index_col=0)

    dataset = pd.concat([df1, df2, df3], axis=0)
    dataset['icu_type'] = dataset['icu_type'].astype(str)

    ignore_features = ['morphine', 'morphine', 'midazolam', 'vasopressin', 'dopamine', 'propofol', 'dobutamine', \
    'epinephrine', 'dexmedetomidine','norepinephrine', 'remifentanil', 'BT_mean', 'BT_std', 'Ventilator_Tidal volume(setting)']
    
    # final dataset
    dataset = dataset.drop(columns=ignore_features, axis=1)
    dataset = dataset.iloc[:, 3: ]

    return label, dataset


    
if __name__ == "__main__":
	main()