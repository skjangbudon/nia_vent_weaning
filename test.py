# Python Script for Testing
import numpy as np, pandas as pd


# 95% CI function

def ci95(inp):
    max95 = round(np.mean(inp) + (1.96 * (np.std(inp) / np.sqrt(len(inp)))),2)
    min95 = round(np.mean(inp) - (1.96 * (np.std(inp) / np.sqrt(len(inp)))),2)
    return min95, max95