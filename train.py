# Python Script for Training

# Preprocessing
import tqdm, numpy as np, warnings, pandas as pd, re, random, datetime
# About Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.externals import joblib    # For Saving Models
# Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# Evaluation
from sklearn.metrics  import roc_auc_score, f1_score, recall_score, confusion_matrix, precision_score, average_precision_score, roc_curve, accuracy_score, auc

warnings.filterwarnings('ignore')


def data_split(dataset, input_window, num_sampling, dst_dir):

    '''
        Input
            1. dataset : Total dataset
            2. input_window : Data Length
            3. num_sampling : Number of Random Sampling
            4. dst_dir : split data destination
    '''

    print('Dataset Sampling Started at ', datetime.datetime.now())

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
    dataset

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
    for s_i in tqdm.tqdm(range(0, num_sampling)):
        print('Sampling index : ', s_i + 1)
        random.seed(s_i)

        train_suc = random.sample(success_group, int(len(success_group)*0.8))
        train_fail = random.sample(fail_group, int(len(fail_group)*0.8))

        # Trainset
        train_pid = train_suc + train_fail

        trainset = dataset[dataset['pid'].isin(train_pid)]
        trainset = trainset.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '_', x))

        testset = dataset[~dataset['pid'].isin(train_pid)]
        testset = testset.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '_', x))

        train_x = trainset.drop(['pid', 'label'], axis=1)
        train_y = trainset['label']

        test_x = testset.drop([ 'pid', 'label'], axis=1)
        test_y = testset['label']

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

        train_x = train_imputed
        test_x = test_imputed

    trainset = pd.concat([train_x, train_y], axis=1)
    testset = pd.concat([test_x, test_y], axis=1)

    trainset.to_csv(dst_dir + 'trainset_' + str(s_i) + '.csv')
    testset.to_csv(dst_dir  + 'testset_' + str(s_i) + '.csv')

    print('Dataset Sampling Ended at ', datetime.datetime.now(), 'Time Elapsed: ', )

    return trainset, testset


def train(train_x, train_y, model, save_dir):

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

    clf.fit(train_x,train_y)
# 객체를 pickled binary file 형태로 저장한다 
file_name = 'object_01.pkl' 
joblib.dump(obj, file_name) 