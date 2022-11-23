# Python Script for Testing
import numpy as np, pandas as pd
import joblib, os
from configparser import ConfigParser
import datetime as dt
from timeit import default_timer as timer
from pytz import timezone
# Evaluation
from module.utils_module import evaluation, get_logger, set_logger
from sklearn.metrics  import confusion_matrix

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

    # for evaluation
    auc_list = []

    # Test dataset
    testset = pd.read_csv(input_data_path + "testset.csv")

    test_x = testset.drop(['pid', 'label'], axis=1)
    test_y = list(testset['label'])
    logger.info('Model Inference Proceeding.... Test Experiment ' + str(s_i))

    # Import model
    clf = joblib.load(model_path + model + '.pkl') 
    # prediction
    y_prob = clf.predict_proba(test_x)[:,1]
    pred_result = evaluation(y_prob, test_y, cut_off=0.5)

    # append result in list
    auc_list.append(pred_result[0])

    # Save Final Result
    testset['y_prob'] = y_prob
    testset['y_pred'] = pred_result[-1]
    testset['Real'] = test_y
    result_df = testset[['pid', 'y_prob', 'y_pred', 'Real']]
    result_df.to_csv(result_path + 'pred_result.csv')

    logger.info('====='*20)
    logger.info('*----- FINAL OUTPUT')
    logger.info('AUROC: ' + str(round(np.mean(auc_list),2)))

    # Confusion Matrix
    logger.info('====='*20)
    logger.info('*----- CREATE CONFUSION MATRIX')
    cm = confusion_matrix(test_y, pred_result[-1])
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(result_path + 'Confusion_Matrix.csv')

    end = timer()

    logger.info('====='*20)
    logger.info('*----- Prediction Ended at ' + f'[ {dt.datetime.now(timezone("Asia/Seoul"))} ]' + '\tTime elapsed: ' + f'[ {dt.timedelta(seconds=end-start)} seconds]')

if __name__ == '__main__':
    model_test()