import sys

import pandas as pd
import numpy as np
import re
import argparse
from Plot.plot_accuracy_vs_epoch import accuracy_plot


def extract_accuracy(neurosim_v, in_log, out_excel):
    """
    Scans for accuracy data from Neurosim log file. Stores it in excel file and creates plot of accuracy vs epoch in plotly
    arguments
    neurosim_v = "MLP" for v3.0 or "DNN" for v2.1
    in_log = input log file
    out_excel = output excel file
    """

    if neurosim_v == "MLP":
        re_exp = "(?<=epochs is : )[\d]{2}\.[\d]{2}"

    if neurosim_v == "DNN":
        re_exp = "(?<=Accuracy: )[\d]{4}"

    # Read text from input file
    with open(in_log) as f:
        txt = f.read()

    # stores list of strings
    matches = re.findall(re_exp, txt)

    # converts it to string of floats
    accuracy = [float(number) for number in matches]

    results = pd.DataFrame({'accuracy': accuracy})
    if neurosim_v == "DNN":
        results = results/100
    results.index.name = 'epoch'
    results.index += 1
    results.to_excel(out_excel)

    accuracy_plot(out_excel, averaged_samples=False, group_size= 20, filetype="xlsx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts Accuracy data from MLP log file and stores in excel sheet')
    parser.add_argument('log_file', help='MLP log file', nargs='?', type= str,
                        default = "data/Apr-23/6bit_with_D2D"
    )
    parser.add_argument('result_excel_file', help='output excel file',nargs='?', type= str,
                        default = "data/Apr-23/6bit_with_D2D.xlsx"
                        )
    parser.add_argument('neurosim_v', help='specify which version of Neurosim that created log file ("DNN/"MLP")',nargs='?', type= str,
                        default = "DNN"
    )
    args = parser.parse_args()

    extract_accuracy(neurosim_v=args.neurosim_v ,in_log=args.log_file, out_excel=args.result_excel_file)
