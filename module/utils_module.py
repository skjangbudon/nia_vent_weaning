# Eval Function
import numpy as np, pandas as pd
import joblib, os
import datetime as dt
import logging

from configparser import ConfigParser
from timeit import default_timer as timer
from sklearn.metrics  import f1_score, recall_score, confusion_matrix, precision_score, average_precision_score, roc_curve, accuracy_score, auc

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

    return roc_score, prc, accuracy, prec, rec, f1, sen, spe, CM, y_pred


# Get Log


def get_logger(name, level=logging.DEBUG, resetlogfile=False, path='log'):
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

    fname = os.path.join(path, name+'.log')
    os.makedirs(path, exist_ok=True) 
    if resetlogfile :
        if os.path.exists(fname):
            os.remove(fname) 
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(fname)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def set_logger(tag, path='.'):
    logger = get_logger(f'{tag}', resetlogfile=True, path=path)
    logger.setLevel(logging.INFO)
    return logger