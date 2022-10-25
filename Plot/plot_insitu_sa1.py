import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

results_path = '../saved_models/0910/results_0910.xlsx'
df_100=pd.read_excel(results_path,sheet_name='50_ep',index_col=0)
#print(df_100['0p'])

#fig = make_subplots(rows=3,cols=1, subplot_titles=('0% Stuck-at-1','1% Stuck-at-1','10% Stuck-at-1'))

x1,x2,x3= df_100['0p'], df_100['1p'], df_100['10p']
hist_data = [x3,x2,x1]
trace_colors = ['magenta','lightsalmon', 'darkcyan']
group_labels = ['10% Stuck-at-1','1% Stuck-at-1','0% Stuck-at-1']

fig2 = ff.create_distplot(
    hist_data,group_labels,
    colors=trace_colors,
    bin_size=0.2,
    show_hist=False
)

fig2.update_xaxes(dict(
    title='Accuracy'
))
fig2.update_yaxes(dict(
    title='Frequency'
))
fig2.update_layout(dict(
    title={
        'text':'Impact of Stuck-at-1 Devices on Accuracy (50 epochs of in-situ training)',
        'x':0.5
    }
))


mean_text_position=['bottom center', 'bottom right', 'top right']

for j in range(len(hist_data)):
    print(j)
    print(hist_data[j].mean())

    fig2.add_trace( go.Scatter(
        x=[hist_data[j].mean()]*2,
        y=[0,0.7],
        showlegend=False,
        #name='mean_' + str(j),
        mode= 'lines+text',
        textposition=mean_text_position[j],
        text=['mean = %.2f'%hist_data[j].mean()],
        textfont={'color': trace_colors[j]},
        line = dict(color=trace_colors[j], dash='dot'),
    ))

    """
               name='mean_' + str(j), text=['dwdc'],
               opacity=0.5, line=dict(color=trace_colors[j], dash='dot'))
    """
        
    """
    fig2.add_shape(type="line",x0=hist_data[j].mean(), x1=hist_data[j].mean(),
                   y0 =0, y1=0.7 , xref='x', yref='y',
                   #line_width=1,
                   name = 'mean_'+str(j),text=['dwdc'],
                   opacity=0.5, line = dict(color = trace_colors[j], dash = 'dot'))
    """

"""
fig.add_trace(go.Histogram(
    name='0%',
    nbinsx=20,
    x=df_100['0p']
),row = 1, col = 1)



fig.add_trace(go.Histogram(
    name='1%',
    nbinsx=20,
    x=df_100['1p']
),row = 2, col = 1)

fig.add_trace(go.Histogram(
    name='10%',
    nbinsx=20,
    x=df_100['10p']
),row = 3, col = 1)

fig.update_layout(
    #xaxis_title_text='Accuracy',
    #yaxis_title_text='Number of Samples',
    bargap=0.2, showlegend=True
)



fig.update_xaxes(range=[87,94], title='Accuracy')
fig.update_yaxes(range=[0,15], title='Samples')



#fig.show()
"""

fig2.show()