import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

results_path = '../saved_models/0910/switching.xlsx'
df = pd.read_excel(results_path)
# remove 1st and 100th epoch
df=df.drop(axis=0, index=[0,99])

# print(df_100['0p'])

# fig = make_subplots(rows=3,cols=1, subplot_titles=('0% Stuck-at-1','1% Stuck-at-1','10% Stuck-at-1'))

epoch_series, sw_pos, sw_neg = df['epoch'], df['-1to1'], df['1to-1']
trace_colors = ['lightsalmon', 'darkcyan']
group_labels = ['-1 to 1', '1 to -1']

fig2 = make_subplots(rows=1,cols=1)

fig2.add_trace(go.Scatter(
    name='-1 to 1',
    x=epoch_series,
    y=sw_pos,
    mode='lines+markers',
    opacity=0.5,
    line = dict(color = trace_colors[0], width=1)
    ),
    row=1,
    col=1
)

fig2.add_trace(go.Scatter(
    name='1 to -1',
    x=epoch_series,
    y=sw_neg,
    mode='lines+markers',
    opacity=0.5,
    line = dict(color = trace_colors[1], width=1)
    ),
    row=1,
    col=1
)


fig2.update_layout(
    showlegend=True,
    title = {
        'text': 'Number of device switches per epoch',
        'x': 0.5
    }
)

fig2.update_xaxes(title='Epoch')
fig2.update_yaxes(title='Number of devices')



fig2.show()