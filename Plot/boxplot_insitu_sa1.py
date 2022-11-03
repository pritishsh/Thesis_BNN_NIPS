import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

if __name__=='__main__':

    results_path = '../saved_models/0910/results_0910.xlsx'
    df_100 = pd.read_excel(results_path, sheet_name='100_ep', index_col=0)
    df_50 = pd.read_excel(results_path, sheet_name='50_ep', index_col=0)

    fig = go.Figure()
    fig.update_layout(
        title={ 'text': 'Impact of Stuck-at-1 Devices on Accuracy',
                'x':    0.5,
                }
    )

    box_data_100=[
        {'name':'0% Stuck-at-1', 'data': df_100['0p'], 'col': 'lightcyan'},
        {'name':'1% Stuck-at-1', 'data': df_100['1p'], 'col': 'coral'},
        {'name': '10% Stuck-at-1', 'data': df_100['10p'], 'col': 'linen'},
    ]

    box_data_final = box_data_100

    fig.update_xaxes(dict(
        type='category'
    ))
    fig.update_yaxes(dict(
        title='Accuracy (%)'
    ))

    scatter_x = [i['name'] for i in box_data_final]
    mean_y = [i['data'].mean() for i in box_data_final]
    error_y = [i['data'].std() for i in box_data_final]
    scatter_text = ['Mean = {:.2f}, Std dev = {:.2f}'.format(i['data'].mean(), i['data'].std() ) for i in box_data_final]

    fig.add_trace(go.Scatter(x=scatter_x,
                             y=mean_y,
                             error_y=dict(type='data', array= error_y),
                             mode='markers+text',
                             marker=dict(
                                 symbol='diamond',
                                 color = 'red',
                             ),
                             text=scatter_text,
                             textposition='middle right',
                             #texttemplate='hh'
                             ))

    fig.add_trace(go.Scatter(y=box_data_final[0]['data'],
                             x=['10% Stuck-at-1']*len(box_data_final[0]['data']),
                             mode='markers',
                             ))

    '''
    for data in box_data_100:
        fig.add_trace(go.Box({'y':data['data'],
                              'fillcolor':data['col'],
                              'name':data['name'],
                              }, boxmean=True,

                             ))
    '''

    '''
    for j in range(len(hist_data)):
        print(j)
        print(hist_data[j].mean())

        fig2.add_trace(go.Scatter(
            x=[hist_data[j].mean()] * 2,
            y=[0, 0.7],
            showlegend=False,
            # name='mean_' + str(j),
            mode='lines+text',
            textposition=mean_text_position[j],
            text=['mean = %.2f' % hist_data[j].mean()],
            textfont={'color': trace_colors[j]},
            line=dict(color=trace_colors[j], dash='dot'),
        ))
    '''


    fig.show()