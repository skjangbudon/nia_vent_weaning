import pandas as pd
import numpy as np
from datetime import timedelta
import tqdm
import warnings

warnings.filterwarnings('ignore')

data_path = '/VOLUME/nia_vent_weaning/data/integrated_data/before/'

# 1
def get_vital(data_path):

    # vital-sign data
    vital = pd.read_csv(data_path + 'vitalsign.csv', index_col=0, parse_dates=['vital_datetime']).iloc[:, :7]
    vital.columns = ['hospital_id', 'patient_id', 'hadm_id', 'charttime', 'item', 'value', 'unit']
    vital = vital[['patient_id', 'charttime', 'item', 'value']]
    vital['patient_id'] = vital['patient_id'].astype(str)

    return vital

# 2
def get_lab(data_path):

    lab = pd.read_csv(data_path + 'lab.csv', index_col=0, parse_dates=['lab_datetime']).iloc[:, :8]
    lab.columns = ['hospital_id', 'patient_id', 'hadm_id', 'charttime', 'lab_code', 'item', 'value', 'unit']
    lab = lab[['patient_id', 'charttime', 'item', 'value']]

    return lab

# 3
def get_vetil_param(data_path):
    # Ventilator parameters 
    ventil_param = pd.read_csv(data_path + 'ventilator_parameter.csv', index_col=0, parse_dates=['parameter_datetime'])
    ventil_param = ventil_param[~ventil_param.parameter.isnull()==True] # Null값 있는 경우 제외

    t_aj = ventil_param[ventil_param['hospital_id']==2]
    t_snu = ventil_param[ventil_param['hospital_id']==1]
    t_ys = ventil_param[ventil_param['hospital_id']==4]

    a = set(t_snu.parameter)
    b = set(t_aj.parameter)
    c = set(t_ys.parameter)

    set1 = a
    set2 = b
    set3 = c

    final_vent_param = list(set1 & set2 & set3)
    ventil_param = ventil_param[ventil_param['parameter'].isin(final_vent_param)]
    ventil_param = ventil_param[['hospital_id', 'patient_id', 'hadm_id', 'mv_id', 'parameter', 'parameter_datetime', 'value', 'valuenum', 'unit']]
    ventil_param.columns = ['hospital_id', 'patient_id', 'hadm_id', 'mv_id', 'item', 'charttime', 'value', 'valuenum', 'unit']
    ventil_param = ventil_param[['patient_id', 'charttime', 'item', 'value']]
    ventil_param['item'] = ventil_param.item.str.split('/').str[1]
    ventil_param['patient_id'] = ventil_param['patient_id'].astype(str)

    ventil_param

    return ventil_param

# 4
def get_height_weight(data_path):
        
    # Height/Weight
    height = pd.read_csv(data_path + 'heightandweight.csv', index_col=0).iloc[:, :7]
    height.columns = ['hospital_id', 'patient_id', 'hadm_id', 'charttime', 'item', 'value', 'unit']
    height = height[['patient_id', 'charttime', 'item', 'value']]

    return height

# 5
def get_diagnosis(data_path): 
        
    # diagnosis
    diag = pd.read_csv(data_path + 'diagnosis.csv', index_col=0).iloc[:, :6]
    diag.columns = ['hospital_id', 'patient_id', 'hadm_id', 'diagnosis_datetime', 'icd10', 'icd10_desc']
    diag = diag[['patient_id', 'diagnosis_datetime', 'icd10']]

    return diag

# 6
def get_drug(data_path):

    drug = pd.read_csv(data_path + 'drug.csv', index_col=0)
    drug = drug[['patient_id', 'drug_datetime', 'drug_name', 'drug_dose']]

    drug = drug[~drug.drug_name.isnull()==True] # Drug_name에서 Null값 있는 경우 제외
    drug.loc[drug.drug_name.str.contains('Norepinephrine'), 'drug_name'] = 'norepinephrine'
    drug.loc[drug.drug_name.str.contains('Epinephrine'), 'drug_name'] = 'epinephrine'
    drug['drug_name'] = drug['drug_name'].str.lower()

    # 변수명 통일
    drug.loc[drug.drug_name.str.contains('dexmedetomidine', case=False), 'drug_name'] = 'dexmedetomidine'
    drug.loc[drug.drug_name.str.contains('dextomine', case=False), 'drug_name'] = 'dexmedetomidine'
    drug.loc[drug.drug_name.str.contains('dobutamine', case=False), 'drug_name'] = 'dobutamine'
    drug.loc[drug.drug_name.str.contains('dopamine', case=False), 'drug_name'] = 'dopamine'
    drug.loc[drug.drug_name.str.contains('midazolam', case=False), 'drug_name'] = 'midazolam'
    drug.loc[drug.drug_name.str.contains('morphine', case=False), 'drug_name'] = 'morphine'
    drug.loc[drug.drug_name.str.contains('vasopressin', case=False), 'drug_name'] = 'vasopressin'
    drug.loc[drug.drug_name.str.contains('propofol', case=False), 'drug_name'] = 'propofol'
    drug.loc[drug.drug_name.str.contains('remifentanil', case=False), 'drug_name'] = 'remifentanil'

    drug_var = ['dexmedetomidine', 'dobutamine', 'dopamine', 'epinephrine', 'midazolam', 'morphine', 'norepinephrine', 'propofol', 'remifentanil', 'vasopressin']
    drug = drug[drug['drug_name'].isin(drug_var)]
    drug['drug_datetime'] = pd.to_datetime(drug['drug_datetime'], errors='coerce')
    drug['patient_id'] = drug['patient_id'].astype(str)

    return drug

# 7
def get_demo(data_path):

    # Demo data 정리
    demo = pd.read_csv(data_path + 'demographics.csv', index_col=0)
    demo['birth'] = demo['birth_date'].str[:4].astype(int)
    demo['adm'] = demo['hosp_adm_date'].str[:4].astype(int)
    demo['age'] = demo['adm'] - demo['birth'] + 1
    demo = demo[['patient_id', 'sex', 'age', 'icu_type', 'hosp_adm_date', 'hosp_discharge_date', 'icu_adm_date', 'icu_discharge_date', 'death_datetime']]
    return demo

# 8
def get_label(data_path):

    demo = get_demo(data_path)
    label_df = pd.read_csv(data_path + 'ventilator.csv', index_col=0)
    label_df['mv_endtime'] = pd.to_datetime(label_df['mv_endtime'])
    label_df['mv_wave_endtime'] = pd.to_datetime(label_df['mv_wave_endtime'])
    label_df = pd.merge(label_df, demo, on='patient_id', how='left')

    return label_df

