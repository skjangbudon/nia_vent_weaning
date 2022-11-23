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

    parser = ConfigParser()
    parser.read('/VOLUME/nia_vent_weaning/config/test_config.ini')
    # set path
    input_data_path = parser.get('PATH', 'input_data_path')
    model_path = parser.get('PATH', 'model_path')
    result_path = parser.get('PATH', 'result_path')
    s_i = int(parser.get('OPTION', 'seed'))
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

    for s_i in range(0, 30):


        # for evaluation
        final_df = pd.DataFrame()
        auc_list = []

        # Test dataset
        testset = pd.read_csv(input_data_path + "testset_" + str(s_i) + ".csv")

        test_x = testset.drop(['pid', 'label']+ignore_list, axis=1)
        test_y = list(testset['label'])
        logger.info('Model Inference Proceeding.... Test Experiment ' + str(s_i))

        # Import model
        clf = joblib.load(model_path + model + '_' + str(s_i) + '.pkl') 
        # prediction
        y_prob = clf.predict_proba(test_x)[:,1]
        pred_result = evaluation(y_prob, test_y, cut_off=0.1)

        # append result in list
        auc_list.append(pred_result[0])

        # confusion_df = pred_result[-2]    # Confusion Matrix Results

        testset['y_prob'] = y_prob
        testset['y_pred'] = pred_result[-1]
        testset['Real'] = test_y
        result_df = testset[['pid', 'y_prob', 'y_pred', 'Real']]
        # result_df.to_csv(result_path + 'model_prob/pred_result_' + str(s_i) + '.csv')

        # Save Final Result
        # final_df['AUROC'] = auc_list
        # final_df.to_csv(result_path + model + '_result.csv')

        end = timer()

        # logger.info('====='*20)
        # logger.info('*----- FINAL OUTPUT')
        logger.info('AUROC: ' + str(round(np.mean(auc_list),2)))


        # Confusion Matrix
        # logger.info('====='*20)
        # logger.info('*----- CREATE CONFUSION MATRIX')
        cm = confusion_matrix(test_y, pred_result[-1])
        cm_df = pd.DataFrame(cm)
        # cm_df.to_csv(result_path + 'Confusion_Matrix.csv')

        # logger.info('====='*20)

        # logger.info('*----- Prediction Ended at ' + f'[ {dt.datetime.now(timezone("Asia/Seoul"))} ]' + '\tTime elapsed: ' + f'[ {dt.timedelta(seconds=end-start)} seconds]')

    return final_df


if __name__ == '__main__':
    model_test()