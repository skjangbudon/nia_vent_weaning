import pandas as pd
import numpy as np
from datetime import timedelta
import tqdm
import warnings

warnings.filterwarnings('ignore')

data_path = '/VOLUME/nia_vent_weaning/data/integrated_data/before/'


def get_vital(data_path):

    # vital-sign data
    vital = pd.read_csv(data_path + 'Vitalsign.csv', index_col=0, parse_dates=['vital_datetime']).iloc[:, :7]
    vital.columns = ['hospital_id', 'patient_id', 'hadm_id', 'charttime', 'item', 'value', 'unit']
    vital = vital[['patient_id', 'charttime', 'item', 'value']]
    vital['patient_id'] = vital['patient_id'].astype(str)

    return vital

def get_lab(data_path):

    lab = pd.read_csv(data_path + 'Lab.csv', index_col=0, parse_dates=['lab_datetime']).iloc[:, :8]
    lab.columns = ['hospital_id', 'patient_id', 'hadm_id', 'charttime', 'lab_code', 'item', 'value', 'unit']
    lab = lab[['patient_id', 'charttime', 'item', 'value']]

    return lab

def get_vetil_param(data_path):
    # Ventilator parameters 
    ventil_param = pd.read_csv(data_path + 'Ventilator-parameter.csv', index_col=0, parse_dates=['parameter_datetime'])
    ventil_param = ventil_param[~ventil_param.parameter.isnull()==True] # Null값 있는 경우 제외

    t_aj = ventil_param[ventil_param['hospital_id']==2]
    t_snu = ventil_param[ventil_param['hospital_id']==1]
    a = set(t_snu.parameter)
    b = set(t_aj.parameter)
    set1 = a
    set2 = b

    final_vent_param = list(set1 & set2)
    ventil_param = ventil_param[ventil_param['parameter'].isin(final_vent_param)]
    ventil_param = ventil_param[['hospital_id', 'patient_id', 'hadm_id', 'mv_id', 'parameter', 'parameter_datetime', 'value', 'valuenum', 'unit']]
    ventil_param.columns = ['hospital_id', 'patient_id', 'hadm_id', 'mv_id', 'item', 'charttime', 'value', 'valuenum', 'unit']
    ventil_param = ventil_param[['patient_id', 'charttime', 'item', 'value']]
    ventil_param['item'] = ventil_param.item.str.split('/').str[1]
    ventil_param['patient_id'] = ventil_param['patient_id'].astype(str)

    ventil_param

    return ventil_param

def get_height_weight(data_path):
        
    # Height/Weight
    height = pd.read_csv(data_path + 'Heightandweight.csv', index_col=0).iloc[:, :7]
    height.columns = ['hospital_id', 'patient_id', 'hadm_id', 'charttime', 'item', 'value', 'unit']
    height = height[['patient_id', 'charttime', 'item', 'value']]

    return height

def get_diagnosis(data_path): 
        
    # diagnosis
    diag = pd.read_csv(data_path + 'Diagnosis.csv', index_col=0).iloc[:, :6]
    diag.columns = ['hospital_id', 'patient_id', 'hadm_id', 'diagnosis_datetime', 'icd10', 'icd10_desc']
    diag = diag[['patient_id', 'diagnosis_datetime', 'icd10']]

    return diag

def get_drug(data_path):

    drug = pd.read_csv(data_path + 'Drug.csv', index_col=0)
    drug = drug[['patient_id', 'drug_datetime', 'drug_name', 'drug_dose']]

    drug = drug[~drug.drug_name.isnull()==True] # Drug_name에서 Null값 있는 경우 제외
    drug.loc[drug.drug_name.str.contains('Norepinephrine'), 'drug_name'] = 'norepinephrine'
    drug.loc[drug.drug_name.str.contains('Epinephrine'), 'drug_name'] = 'epinephrine'
    drug['drug_name'] = drug['drug_name'].str.lower()

    # 변수명 통일
    drug.loc[drug.drug_name.str.contains('dexmedetomidine'), 'drug_name'] = 'dexmedetomidine'
    drug.loc[drug.drug_name.str.contains('dextomine'), 'drug_name'] = 'dexmedetomidine'
    drug.loc[drug.drug_name.str.contains('dobutamine'), 'drug_name'] = 'dobutamine'
    drug.loc[drug.drug_name.str.contains('dopamine'), 'drug_name'] = 'dopamine'
    drug.loc[drug.drug_name.str.contains('midazolam'), 'drug_name'] = 'midazolam'
    drug.loc[drug.drug_name.str.contains('morphine'), 'drug_name'] = 'morphine'
    drug.loc[drug.drug_name.str.contains('vasopressin'), 'drug_name'] = 'vasopressin'
    drug.loc[drug.drug_name.str.contains('propofol'), 'drug_name'] = 'propofol'
    drug.loc[drug.drug_name.str.contains('remifentanil'), 'drug_name'] = 'remifentanil'

    drug_var = ['dexmedetomidine', 'dobutamine', 'dopamine', 'epinephrine', 'midazolam', 'morphine', 'norepinephrine', 'propofol', 'remifentanil', 'vasopressin']
    drug = drug[drug['drug_name'].isin(drug_var)]
    drug['drug_datetime'] = pd.to_datetime(drug['drug_datetime'], errors='coerce')
    drug['patient_id'] = drug['patient_id'].astype(str)

    return drug

def get_demo(data_path):

    # Demo data 정리
    demo = pd.read_csv(data_path + 'Demographic.csv', index_col=0)
    demo['birth'] = demo['birth_date'].str[:4]
    demo['adm'] = demo['hosp_adm_date'].str[:4]
    demo['age'] = demo['adm'].astype(int) - demo['birth'].astype(int) + 1
    demo = demo[['patient_id', 'sex', 'age', 'icu_type', 'hosp_adm_date', 'hosp_discharge_date', 'icu_adm_date', 'icu_discharge_date', 'death_datetime']]
    return demo



def get_label(data_path):

    demo = get_demo(data_path)
    label_df = pd.read_csv(data_path + 'label.csv', index_col=0)
    label_df['mv_endtime'] = pd.to_datetime(label_df['mv_endtime'])
    label_df['mv_wave_endtime'] = pd.to_datetime(label_df['mv_wave_endtime'])
    label_df = pd.merge(label_df, demo, on='patient_id', how='left')

    return label_df

