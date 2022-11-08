# Python Script for Testing
import numpy as np, pandas as pd
import joblib
from configparser import ConfigParser
import datetime as dt
from timeit import default_timer as timer
from pytz import timezone
# Evaluation
from module.utils_module import *

def model_test():

    # for evaluation
    final_df = pd.DataFrame()
    auc_list = []
    prc_list = []
    acc_list = []
    pre_list = []
    rec_list = []
    f1_list = []
    sen_list = []
    spe_list = []

    parser = ConfigParser()
    parser.read('/VOLUME/nia_vent_weaning/config/test_config.ini')
    # set path
    input_data_path = parser.get('PATH', 'input_data_path')
    model_path = parser.get('PATH', 'model_path')
    result_path = parser.get('PATH', 'result_path')
    num_sampling = int(parser.get('OPTION', 'num_sampling'))
    # set model
    model = parser.get('OPTION', 'model')

    logger = set_logger('test', path=result_path)
    logger.info('*----- SET CONFIGS')
    logger.info('RESULT_PATH : ' + result_path)
    logger.info('MODEL_PATH : ' + model_path)
    logger.info('ML_MODEL : ' + model)
    logger.info('====='*20)
    logger.info('*----- Prediction Started at ' + f'[ {dt.datetime.now(timezone("Asia/Seoul"))} ]')
    logger.info('====='*20)

    # 실행 명령어 출력
    logger.info('*----- Python Script command :')
    logger.info(f'{os.path.basename(__file__)}')
    logger.info('====='*20)
    start = timer()

    # Test dataset

    for model_idx in range(num_sampling):

        testset = pd.read_csv(input_data_path + "testset_" + str(model_idx) + ".csv")
        test_x = testset.drop(['pid', 'label'], axis=1)
        test_y = list(testset['label'])
        logger.info('Model Inference Proceeding.... Test Experiment ' + str(model_idx+1))

        # Import model
        clf = joblib.load(model_path + model + '_' + str(model_idx) + '.pkl') 
        # prediction
        y_prob = clf.predict_proba(test_x)[:,1]

        pred_result = evaluation(y_prob, test_y, cut_off=0.1)

        # append result in list
        auc_list.append(pred_result[0])
        # prc_list.append(pred_result[1])
        # acc_list.append(pred_result[2])
        # pre_list.append(pred_result[3])
        # rec_list.append(pred_result[4])
        # f1_list.append(pred_result[5])
        # sen_list.append(pred_result[6])
        # spe_list.append(pred_result[7])

        # confusion_df = pred_result[-2]    # Confusion Matrix Results

        testset['y_prob'] = y_prob
        testset['y_pred'] = pred_result[-1]
        testset['Real'] = test_y
        result_df = testset[['pid', 'y_prob', 'y_pred', 'Real']]
        result_df.to_csv(result_path + 'model_prob/pred_result' + str(model_idx) + '.csv')

    # Save Final Result
    final_df['AUROC'] = auc_list
    # final_df['AUPRC'] = prc_list
    # final_df['Accuracy'] = acc_list
    # final_df['Precision'] = pre_list
    # final_df['Recall'] = rec_list
    # final_df['F1_score'] = f1_list
    # final_df['Sensitivity'] = sen_list
    # final_df['Specificity'] = spe_list
    final_df.to_csv(result_path + model + '_result.csv')
   
    end = timer()

    logger.info('====='*20)
    logger.info('*----- FINAL OUTPUT')
    logger.info('AUROC: ' + str(round(np.mean(auc_list),2)) + '(' + str(ci95(auc_list)[0]) + '-' + str(ci95(auc_list)[1]) + ')')
    # logger.info('AUPRC: ' + str(round(np.mean(prc_list),2)) + '(' + str(ci95(prc_list)[0]) + '-' + str(ci95(prc_list)[1]) + ')')
    # logger.info('Accuracy: ' + str(round(np.mean(acc_list),2)) + '(' + str(ci95(acc_list)[0]) + '-' + str(ci95(acc_list)[1]) + ')')
    # logger.info('Precision: ' + str(round(np.mean(pre_list),2)) + '(' + str(ci95(pre_list)[0]) + '-' + str(ci95(pre_list)[1]) + ')')
    # logger.info('Recall: ' + str(round(np.mean(rec_list),2)) + '(' + str(ci95(rec_list)[0]) + '-' + str(ci95(rec_list)[1]) + ')')
    # logger.info('F1-score: ' + str(round(np.mean(f1_list),2)) + '(' + str(ci95(f1_list)[0]) + '-' + str(ci95(f1_list)[1]) + ')')
    # logger.info('Sensitivity: ' + str(round(np.mean(sen_list),2)) + '(' + str(ci95(sen_list)[0]) + '-' + str(ci95(sen_list)[1]) + ')')
    # logger.info('Spesificity: ' + str(round(np.mean(spe_list),2)) + '(' + str(ci95(spe_list)[0]) + '-' + str(ci95(spe_list)[1]) + ')')

    logger.info('====='*20)

    logger.info('*----- Prediction Ended at ' + f'[ {dt.datetime.now(timezone("Asia/Seoul"))} ]' + '\tTime elapsed: ' + f'[ {dt.timedelta(seconds=end-start)} seconds]')

    return final_df


if __name__ == '__main__':
    model_test()