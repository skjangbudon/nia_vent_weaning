# Python Script for Testing
import numpy as np, pandas as pd
import joblib
from configparser import ConfigParser
import datetime as dt
from timeit import default_timer as timer
# Evaluation
from sklearn.metrics  import roc_auc_score, f1_score, recall_score, confusion_matrix, precision_score, average_precision_score, roc_curve, accuracy_score, auc


# 95% CI function
def ci95(inp):
    max95 = round(np.mean(inp) + (1.96 * (np.std(inp) / np.sqrt(len(inp)))),2)
    min95 = round(np.mean(inp) - (1.96 * (np.std(inp) / np.sqrt(len(inp)))),2)
    return min95, max95

# Calculate Evaluation Indicator
def evaluation(y_prob, test_y, cut_off):

    pred_positive_label = y_prob

    # AUROC
    fprs, tprs, threshold = roc_curve(test_y, pred_positive_label)
    y_pred = np.where(y_prob > cut_off, 1, 0)

    roc_score = auc(fprs, tprs)
    prc = average_precision_score(test_y, pred_positive_label)
    accuracy = accuracy_score(test_y, y_pred)
    prec = precision_score(test_y, y_pred)
    rec = recall_score(test_y, y_pred)
    f1 = f1_score(test_y, y_pred)
    CM = confusion_matrix(test_y, y_pred)
    TN, FN, TP, FP = CM[0][0], CM[1][0], CM[1][1], CM[0][1]
    sen = TP/(TP+FN)
    spe = TN/(TN+FP)

    return roc_score, prc, accuracy, prec, rec, f1, sen, spe

def model_test():
    start = timer()
    print('')
    print('Prediction Started at ', dt.datetime.now())
    print('====='*20)
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

    # Test dataset

    for model_idx in range(num_sampling):
        tmp_df = pd.DataFrame()
        testset = pd.read_csv(input_data_path + "testset_" + str(model_idx) + ".csv")
        test_x = testset.drop(['label'], axis=1)
        test_y = list(testset['label'])

        # Import model
        clf = joblib.load(model_path + model + '_' + str(model_idx) + '.pkl') 
        # prediction
        y_prob = clf.predict_proba(test_x)[:,1]

        pred_result = evaluation(y_prob, test_y, cut_off=0.1)

        # append result in list
        auc_list.append(pred_result[0])
        prc_list.append(pred_result[1])
        acc_list.append(pred_result[2])
        pre_list.append(pred_result[3])
        rec_list.append(pred_result[4])
        f1_list.append(pred_result[5])
        sen_list.append(pred_result[6])
        spe_list.append(pred_result[7])

        tmp_df['Prediction'] = y_prob
        tmp_df['Real'] = test_y
        tmp_df.to_csv(result_path + 'model_prob/pred_result' + str(model_idx) + '.csv')

    # Save Final Result
    final_df['AUROC'] = auc_list
    final_df['AUPRC'] = prc_list
    final_df['Accuracy'] = acc_list
    final_df['Precision'] = pre_list
    final_df['Recall'] = rec_list
    final_df['F1_score'] = f1_list
    final_df['Sensitivity'] = sen_list
    final_df['Specificity'] = spe_list
    final_df.to_csv(result_path + model + '_result.csv')
    

    print('AUROC: ', str(round(np.mean(auc_list),2)) + '(' + str(ci95(auc_list)[0]) + '-' + str(ci95(auc_list)[1]) + ')')
    print('AUPRC: ',  str(round(np.mean(prc_list),2)) + '(' + str(ci95(prc_list)[0]) + '-' + str(ci95(prc_list)[1]) + ')')
    print('Accuracy: ',  str(round(np.mean(acc_list),2)) + '(' + str(ci95(acc_list)[0]) + '-' + str(ci95(acc_list)[1]) + ')')
    print('Precision: ',  str(round(np.mean(pre_list),2)) + '(' + str(ci95(pre_list)[0]) + '-' + str(ci95(pre_list)[1]) + ')')
    print('Recall: ',  str(round(np.mean(rec_list),2)) + '(' + str(ci95(rec_list)[0]) + '-' + str(ci95(rec_list)[1]) + ')')
    print('F1-score: ',  str(round(np.mean(f1_list),2)) + '(' + str(ci95(f1_list)[0]) + '-' + str(ci95(f1_list)[1]) + ')')
    print('Sensitivity: ', str(round(np.mean(sen_list),2)) + '(' + str(ci95(sen_list)[0]) + '-' + str(ci95(sen_list)[1]) + ')')
    print('Spesificity: ', str(round(np.mean(spe_list),2)) + '(' + str(ci95(spe_list)[0]) + '-' + str(ci95(spe_list)[1]) + ')')
    end = timer()
    print('====='*20)
    print('Prediction Ended at ', dt.datetime.now(), '\tTime elapsed: ', dt.timedelta(seconds=end-start), 'seconds')

    return final_df


if __name__ == '__main__':
    model_test()