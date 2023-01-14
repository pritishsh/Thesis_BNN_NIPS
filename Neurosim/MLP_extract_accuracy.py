import sys

import pandas as pd
import numpy as np
import re
import argparse
from Plot.plot_accuracy_vs_epoch import accuracy_plot


def extract_accuracy(in_log, out_excel):
    '''
    Scans for accuracy data from Neurosim log file. Stores it in excel file and creates plot of accuracy vs epoch in plotly
    arguments:
    in_log = input log file
    out_excel = output excel file
    '''

    re_exp = "(?<=epochs is : )[\d]{2}\.[\d]{2}"

    # Read text from input file
    with open(in_log) as f:
        txt = f.read()

    # stores list of strings
    matches = re.findall(re_exp, txt)

    # converts it to string of floats
    accuracy = [float(number) for number in matches]

    results = pd.DataFrame({'Accuracy': accuracy})
    results.index.name = 'Epoch'
    results.index += 1
    results.to_excel(out_excel)

    accuracy_plot(out_excel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts Accuracy data from MLP log file and stores in excel sheet')
    parser.add_argument('log_file', help='MLP log file', nargs='?', type= str,
                        default = "data/128hl_src_is_500ltpltd.txt"
    )
    parser.add_argument('result_excel_file', help='output excel file',nargs='?', type= str,
                        default = "data/128_src_is_500ltpltd.xlsx"
    )
    args = parser.parse_args()

    extract_accuracy(in_log=args.log_file, out_excel=args.result_excel_file)
