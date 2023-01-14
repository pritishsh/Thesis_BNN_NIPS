import pandas as pd

results_path = '../saved_models/0910/results_0910.xlsx'
df_150 = pd.read_excel(results_path, sheet_name='150_ep', index_col=0)
df_100 = pd.read_excel(results_path, sheet_name='100_ep', index_col=0)
df_50 = pd.read_excel(results_path, sheet_name='50_ep', index_col=0)


box_data_50 = [
    {'name': '0% Stuck-at-1', 'data': df_50['0p'], 'col': 'lightcyan'},
    {'name': '1% Stuck-at-1', 'data': df_50['1p'], 'col': 'coral'},
    {'name': '10% Stuck-at-1', 'data': df_50['10p'], 'col': 'linen'},
]

box_data_100_in = [
    {'name': '0% Stuck-at-1', 'data': df_100['0p'], 'col': 'lightcyan'},
    {'name': '1% Stuck-at-1 (in-situ)', 'data': df_100['1p'], 'col': 'coral'},
    {'name': '10% Stuck-at-1 (in-situ)', 'data': df_100['10p'], 'col': 'linen'},
]

box_data_100_ex = [
    {'name': '0% Stuck-at-1', 'data': df_100['0p'], 'col': 'lightcyan'},
    {'name': '1% Stuck-at-1 (ex-situ)', 'data': df_100['1p_ex'], 'col': 'coral'},
    {'name': '10% Stuck-at-1 (ex-situ)', 'data': df_100['10p_ex'], 'col': 'linen'},
]

box_data_150_in = [
    {'name': '0% Stuck-at-1', 'data': df_150['0p'], 'col': 'lightcyan'},
    {'name': '1% Stuck-at-1 (in-situ)', 'data': df_150['1p'], 'col': 'coral'},
    {'name': '10% Stuck-at-1 (in-situ)', 'data': df_150['10p'], 'col': 'linen'},
]
box_data_150_ex = [
    {'name': '0% Stuck-at-1', 'data': df_150['0p'], 'col': 'lightcyan'},
    {'name': '1% Stuck-at-1 (ex-situ)', 'data': df_150['1p_ex'], 'col': 'coral'},
    {'name': '10% Stuck-at-1 (ex-situ)', 'data': df_150['10p_ex'], 'col': 'linen'},
]
