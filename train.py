# Python Script for Training

# Preprocessing
import tqdm, os, numpy as np, warnings, pandas as pd, re, random, datetime as dt
from timeit import default_timer as timer
from configparser import ConfigParser
# About Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import joblib    # For Saving Models
# Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

warnings.filterwarnings('ignore')

def data_split():
    '''
        Input
            1. dataset : Total dataset
            2. input_window : Data Length
            3. num_sampling : Number of Random Sampling
            4. dst_dir : split data destination
    '''
    parser = ConfigParser()
    parser.read('/VOLUME/nia_vent_weaning/config/train_config.ini')
    data_dir = parser.get('PATH', 'data_path')
    input_data_path = parser.get('PATH', 'input_data_path')
    s_i = int(parser.get('OPTION', 'num_sampling'))
    input_window = parser.get('OPTION', 'input_length')

    start = timer()

    print('Dataset Sampling Started at ', dt.datetime.now())

    data_dir = '/VOLUME/nia_vent_weaning/data/model_data/' + str(input_window) + 'h/'
    df1 = pd.read_csv(data_dir + '0h_data.csv', index_col=0)
    df2 = pd.read_csv(data_dir + '1h_data.csv', index_col=0)
    df3 = pd.read_csv(data_dir + '2h_data.csv', index_col=0)

    dataset = pd.concat([df1, df2, df3], axis=0)
    dataset['icu_type'] = dataset['icu_type'].astype(str)

    ignore_features = ['morphine', 'morphine', 'midazolam', 'vasopressin', 'dopamine', 'propofol', 'dobutamine', \
    'epinephrine', 'dexmedetomidine','norepinephrine', 'remifentanil', 'BT_mean', 'BT_std', 'Ventilator_Tidal volume(setting)']
    # dataset = dataset.fillna(1111)
    dataset = dataset.drop(columns=ignore_features, axis=1)

    # 불필요한 컬럼 제거
    dataset = dataset.iloc[:, 3: ]

    # Dummy Variables
    icu_cat_df = pd.get_dummies(dataset['icu_type'])
    vt_set_df = pd.get_dummies(dataset['Ventilator mode(setting)'])
    dataset = dataset.drop(columns=['icu_type','Ventilator mode(setting)'], axis=1)
    dataset = pd.concat([dataset, icu_cat_df,vt_set_df], axis=1)

    # Set Case and Control group
    success_group = list(set(dataset[dataset['label']==0]['pid']))
    fail_group = list(set(dataset[dataset['label']==1]['pid']))
    
    # Random Sampling
    print('Sampling index : ', s_i + 1)
    random.seed(s_i)

    train_suc = random.sample(success_group, int(len(success_group)*0.8))
    train_fail = random.sample(fail_group, int(len(fail_group)*0.8))

    # Trainset
    train_pid = train_suc + train_fail

    trainset = dataset[dataset['pid'].isin(train_pid)].reset_index(drop=True)
    trainset = trainset.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '_', x))

    testset = dataset[~dataset['pid'].isin(train_pid)].reset_index(drop=True)
    testset = testset.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '_', x))

    train_x = trainset.drop(['pid', 'label'], axis=1)
    train_y = trainset['label']

    test_x = testset.drop([ 'pid', 'label'], axis=1)
    test_y = testset['label']

    # pid info
    train_list = trainset['pid']
    test_list = testset['pid']

    # MICE
    imp = IterativeImputer(max_iter=30, random_state=0, min_value = 0)

    # TRAIN imputation
    data = train_x
    feature = train_x.keys()
    imp.fit(data)
    data_r = imp.transform(data)
    data_imputed = pd.DataFrame(data_r)
    data_imputed.columns = feature
    train_imputed = data_imputed

    # TEST imputation
    data = test_x
    feature = test_x.keys()
    imp.fit(data)
    data_r = imp.transform(data)
    data_imputed = pd.DataFrame(data_r)
    data_imputed.columns = feature
    test_imputed = data_imputed

    # reset index
    train_x = train_imputed.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    test_x = test_imputed.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)

    final_train = pd.concat([train_x, train_y], axis=1)
    final_test = pd.concat([test_x, test_y], axis=1)   
    print(final_train.keys())
    # add pid
    final_train['pid'] = train_list
    final_test['pid'] = test_list
    
    final_train.to_csv(input_data_path + "trainset_" + str(s_i) + ".csv")
    final_test.to_csv(input_data_path  + "testset_" + str(s_i) + ".csv")
    
    end = timer()
    print('Dataset Sampling Ended at ', dt.datetime.now(), '\tTime elapsed: ', dt.timedelta(seconds=end-start), 'seconds')

    return final_train, final_test


def model_train(train_x, train_y, model,  model_idx):
    start = timer()
    print('Model Training Started at ', dt.datetime.now())

    parser = ConfigParser()
    parser.read('/VOLUME/nia_vent_weaning/config/train_config.ini')
    model_path = parser.get('PATH', 'model_path')

    '''
        Input
            1. dataset : total dataset
            2. model : LR(Logistic Regression), RF(Random Forest), lgbm(lightGBM), SVM(Support Vector Machine)
    '''

    if model == 'LR':
        clf = LogisticRegression(C=1, penalty='l1', solver='liblinear')
    
    elif model == 'RF':
        clf = RandomForestClassifier(n_estimators=150, max_depth=11,random_state=0)
    
    elif model == 'lgbm':
        clf = LGBMClassifier(n_estimators=1500, num_leaves=31, boosting_type='gbdt', metric='binary_logloss',
                        learning_rate=0.01, objective='binary')

    elif model == 'SVM':
        clf = SVC(C=2, gamma=100, kernel='poly', probability=True)

    clf.fit(train_x, train_y)

    # Save Model (pickled binary file)
    file_name = model + '_' + str(model_idx) +'.pkl' 
    joblib.dump(clf, model_path + file_name)

    end = timer()
    print('Model Training Ended at ', dt.datetime.now(), '\tTime elapsed: ', dt.timedelta(seconds=end-start), 'seconds')


if __name__ == '__main__':
    parser = ConfigParser()
    parser.read('/VOLUME/nia_vent_weaning/config/train_config.ini')
    input_data_path = parser.get('PATH', 'input_data_path')
    s_i = int(parser.get('OPTION', 'num_sampling'))
    model =  parser.get('OPTION', 'model')

    if len(os.listdir(input_data_path))==0: # input 데이터가 없으면
        data_split()

        trainset = pd.read_csv(input_data_path + "trainset_" + str(s_i) + ".csv")
        # testset = pd.read_csv(input_data_path + "testset_" + str(model_idx) + ".csv")

        train_x = trainset.drop(['pid', 'label'], axis=1)
        train_y = trainset[['label']]

        model_train(train_x, train_y, model, s_i)

    else:

        trainset = pd.read_csv(input_data_path + "trainset_" + str(s_i) + ".csv")
        # testset = pd.read_csv(input_data_path + "testset_" + str(model_idx) + ".csv")

        train_x = trainset.drop(['label'], axis=1)
        train_y = trainset[['label']]

        model_train(train_x, train_y, model, s_i)
