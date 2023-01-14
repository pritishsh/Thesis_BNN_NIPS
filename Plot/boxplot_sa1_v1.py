import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from plot_formatting import *

if __name__=='__main__':

    box_width = 20


    fig = go.Figure()
    fig.update_layout(
        title={ 'text': 'Impact of Stuck-at-1 Devices on Accuracy (150 Epochs)',
                'x':    0.5,
                },
        showlegend=False,
    )

    box_data_final = box_data_50

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

    '''
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
    '''


    for data in box_data_final:
        data.update({'mean':data['data'].mean(),
                     'std': data['data'].std(),
                     'up_sigma': data['data'].mean() + data['data'].std(),
                     'dn_sigma': data['data'].mean() - data['data'].std(),
                     })
        fig.add_trace(go.Scatter({'y':data['data'],
                                  'x': [data['name']]*len(data['data']),
                                  'fillcolor':data['col'],
                                  #'name':data['name'],
                              },
                                 mode='markers',

                             ))
        fig.add_shape(
            type='rect',
            xanchor=data['name'],
            xsizemode='pixel',
            x0=-box_width,
            x1=box_width,
            y0=data['up_sigma'],
            y1=data['dn_sigma'],
            # height=20,
            # width=5,
        )

        fig.add_shape(
            type='line',
            xanchor=data['name'],
            xsizemode='pixel',
            x0=-2*box_width,
            x1=2*box_width,
            y0=data['mean'],
            y1=data['mean'],
            line={'dash':'dash'}
            #
        )

        fig.add_annotation(
            y=data['mean'],
            x=data['name'],
            xanchor='left',
            yanchor='bottom',
            text="Mean = {:.2f}%".format(float(data['mean'].squeeze())),
            showarrow=False,
            xshift=35)

        fig.add_annotation(
            y=data['mean'],
            x=data['name'],
            xanchor='left',
            yanchor='top',
            text="Std Deviation = {:.2f}".format(float(data['std'].squeeze())),
            showarrow=False,
            xshift=33)

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