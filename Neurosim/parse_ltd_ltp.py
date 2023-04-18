import pandas as pd
import logging
import numpy as np


def parse_excel(excel_sheet, num_ltp, num_ltd):
    total = num_ltp + num_ltp
    logging.basicConfig(filename='data/parsed_ltp_ltd.txt', level=logging.INFO, format='%(message)s')
    xls = pd.ExcelFile(excel_sheet)
    # each element in dictionary is pandas dataframe correspoding to sheet in xlsx file
    df_dict={}
    colnames = ['conductance']
    for sheet_name in xls.sheet_names:
        temp_df = pd.read_excel(excel_sheet, sheet_name= sheet_name, index_col=0, names=colnames, header=None)
        #df_dict.update({sheet_name:temp_df})
        #print(temp_df.columns)
        logging.info('')
        logging.info(sheet_name+'------'*3)
        logging.info('')
        logging.info('exp_LTP_raw=[')

        for i in range(1,num_ltp+1):
            logstring = '{num:2d}, {val:e}'.format(num=i-1 , val=temp_df.loc[i].squeeze())
            logging.info(logstring)

        logging.info('];')
        #logging.info('')
        logging.info('exp_LTD_raw=[')
        for i in range(1,num_ltd+1):
            logstring = '{num:2d}, {val:e}'.format(num=i-1 , val=temp_df.iloc[ - i].squeeze())
            logging.info(logstring)
        logging.info('];')


if __name__ == '__main__':
    x = np.dtype.str
    print(x)
    parse_excel( excel_sheet= "data/48 devices.xlsx",
                 num_ltp=50,
                 num_ltd=60
                 )