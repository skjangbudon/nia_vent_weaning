# Preprocessing Python Script
import pandas as pd
from configparser import ConfigParser
from module.dataset import *

# Integration Total dataset

def get_final_input(data_path, input_window):
    
    # Set Config
    parser = ConfigParser()
    parser.read('/VOLUME/nia_vent_weaning/config/train_config.ini')
    data_path = parser.get('PATH', 'data_path')
    dst_path = parser.get('PATH', 'dst_path')
    input_window = parser.get('OPTION', 'input_length')

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
        final_df.to_csv(dst_path + 'model_data/' + str(input_window) + 'h/' + str(before_time)  + 'h_data.csv')

    return final_df


def preprocessing(input_window):

    '''
        Return : (i) Labelling DataFrame  (ii) dataset DataFrame
    '''

    parser = ConfigParser()
    parser.read('/VOLUME/nia_vent_weaning/config/preprocessing_config.ini')
    data_path = parser.get('PATH', 'data_path')
    input_window = parser.get('OPTION', 'input_length')

    label = dataset.get_label(data_path + 'integrated_data/before/')
    data_dir = data_path + 'model_data/' + str(input_window) + 'h/'
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

    get_final_input()
    # preprocessing()