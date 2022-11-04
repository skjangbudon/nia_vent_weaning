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


def get_final_input(data_path, input_window):

    # import data
    label_df = get_label(data_path)
    vital = get_vital(data_path)
    lab = get_lab(data_path)
    height = get_height_weight(data_path)
    ventil_param = get_vetil_param(data_path)
    drug = get_drug(data_path)

    # # numeric 변수 4개 붙이기
    data_df = pd.concat([vital, lab], axis=0)
    data_df = pd.concat([data_df, height], axis=0)
    data_df['charttime'] = pd.to_datetime(data_df['charttime'], errors='coerce')
    data_df.loc[data_df.item=='몸무게', 'item'] = 'weight'
    data_df.loc[data_df.item=='키', 'item'] = 'height'
    data_df['patient_id'] = data_df['patient_id'].astype(str)
    data_df = data_df[data_df['value'].isna() == False]

    # Variable list
    fixed_var = ['age', 'sex', 'weight', 'height']  # Label_df : Age, Sex, / Data_df : weight/height 
    cat_var = ['icu_type']  

    # Vital + LAB + Drug + Ventilator-parameters
    numeric_var = list(set(vital.item)) + list(set(lab.item))
    ventil_var= list(set(ventil_param.item))[:-1]
    drug_var = list(set(drug.drug_name))

    for before_time in range(0, 3):

        results = dict()
        results['vent_end'] = []
        results['st_time'] = []
        results['end_time'] = []
        results['pid'] = []
        results['label'] = []

        # Fixed Variables
        for i in fixed_var + cat_var + drug_var:
            results[i] = []

        # Vital-sign + LAB
        for i in numeric_var:
            results[i + '_mean'] = []
            results[i + '_std'] = []

        # Ventilator parameters
        for i in ventil_var:
            if i == 'Ventilator_I:E(setting)':
                continue
            else:
                results[i] = []

        for idx in tqdm.tqdm(range(len(label_df))):

            pid = str(label_df.patient_id[idx])
            results['pid'].append(pid)    # add pid
            results['label'].append(label_df.re_mv_status[idx])    # add label

            results['vent_end'].append(label_df.mv_endtime[idx])
            input_end = label_df.mv_endtime[idx] - timedelta(hours=before_time)
            input_st = input_end - timedelta(hours=input_window)

            results['st_time'].append(input_st)
            results['end_time'].append(input_end)

            # Temp data
            tmp_df = data_df[data_df['patient_id']==str(pid)]

            # fixed data
            results['age'].append(label_df.age[idx])    # add age
            results['sex'].append(label_df.sex[idx])    # add sex

            # cat feature
            results['icu_type'].append(label_df.icu_type[idx]) # icu type

            # numeric feature
            tmp_df_num = tmp_df[(tmp_df['charttime']>=input_st) & (tmp_df['charttime']<=input_end)] # 이벤트 발생 1시간 전 시점부터 12시간동안 데이터 input
            tmp_df_num['value'] = pd.to_numeric(tmp_df_num['value'], errors = 'coerce')
            # print(pid, len(tmp_df_num), input_st, input_end)
            # height_df
            height_df = tmp_df[tmp_df.item=='height']

            if len(height_df) == 0:
                height = np.nan
            else:
                height = round(np.mean(list(height_df['value'])),2)

            # weight_df
            weight_df = tmp_df[tmp_df.item=='weight']

            if len(weight_df) == 0:
                weight = np.nan
            else:
                weight = round(np.mean(list(weight_df['value'])),2)

            results['height'].append(height)
            results['weight'].append(weight)

            for nm in numeric_var:
                
                num_val = tmp_df_num[tmp_df_num.item==nm]

                if len(num_val)==0:
                    val_rst_mean = np.nan
                    val_rst_std = np.nan

                else:
                    data_list = list(num_val['value'])

                    rtn_list= []
                    for _tmp in data_list:
                        try:
                            float(_tmp)
                            rtn_list.append(_tmp)
                        except ValueError:
                            pass

                    val_rst_mean = round(np.mean(rtn_list),3)
                    val_rst_std = round(np.std(rtn_list),3)

                results[nm + '_mean'].append(val_rst_mean)
                results[nm + '_std'].append(val_rst_std)

            # Drug data
            drug_tmp = drug[drug.patient_id==str(pid)]
            drug_tmp = drug_tmp[(drug_tmp['drug_datetime']>=input_st) & (drug_tmp['drug_datetime']<=input_end) ]
            drug_tmp['drug_dose'] = pd.to_numeric(drug_tmp['drug_dose'], errors = 'coerce')

            for dg in drug_var:
                drug_val = drug_tmp[drug_tmp.drug_name==dg]

                if len(drug_val)==0:
                    dg_val = np.nan
                else:
                    data_list = list(drug_val['drug_dose'])
                    dg_val = round(np.mean(data_list),3)
                
                results[dg].append(dg_val)

            # Ventilator-parameter data
            vt_tmp = ventil_param[ventil_param['patient_id']==str(pid)]
            vt_tmp = vt_tmp[(vt_tmp['charttime']>=input_st) & (vt_tmp['charttime']<=input_end)].sort_values(['charttime']) # 이벤트 발생 1시간 전 시점부터 12시간동안 데이터 input
            
            for vt in ventil_var:

                if vt == 'Ventilator_I:E(setting)':
                    continue

                elif vt == 'Ventilator mode(setting)':
                    tmp_list = list(vt_tmp[vt_tmp['item']==vt]['value'])

                    if len(tmp_list) == 0:
                        vt_rst = 'etc'
                    else:
                        vt_rst = tmp_list[-1]
                    results[vt].append(vt_rst)

                else:
                    vt_tmp2 = vt_tmp.copy()
                    vt_tmp2['value'] = pd.to_numeric(vt_tmp2['value'], errors='coerce')
                    tmp_list2 = list(vt_tmp2[vt_tmp2['item']==vt]['value'])

                    if len(tmp_list2) == 0:
                        vt_rst = np.nan
                    else:

                        if vt == 'Ventilator_Tidal volume(setting)':
                            vt_rst = round(np.mean(tmp_list2),3)
                            if vt_rst < 0:  # L 단위 시, x 1000
                                vt_rst = vt_rst * 1000
                        else:
                            vt_rst = round(np.mean(tmp_list2),3)

                    results[vt].append(vt_rst)
            
        final_df = pd.DataFrame.from_dict(results)

    return final_df