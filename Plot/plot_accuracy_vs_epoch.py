import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

if __name__=='__main__':
    #results_path = '../saved_models/8000_epochs.xlsx'
    results_path = '../Neurosim/data/200h_1250ep.xlsx'
    #group_labels = ['Accuracy', '1 to -1']

def accuracy_plot(accuracy_data):
    "accuracy data is path to excel sheet containing columns: Accuracy, Epoch"

    df = pd.read_excel(accuracy_data)
    epoch_series, accuracy_series = df['Epoch'], df['Accuracy']
    trace_colors = ['lightsalmon', 'darkcyan']
    fig2 = make_subplots(rows=1,cols=1)

    fig2.add_trace(go.Scatter(
        name='Accuracy',
        x=epoch_series,
        y=accuracy_series,
        #mode='lines+markers',
        mode='markers',
        opacity=1,
        line = dict(color = trace_colors[1], width=1),
        marker = dict(color=trace_colors[1], size=5),
    ),
        row=1,
        col=1
    )

    fig2.update_layout(
        showlegend=True,
        title = {
            'text': 'Accuracy v/s Epoch',
            'x': 0.5
        }
    )

    fig2.update_xaxes(title='Epoch')
    fig2.update_yaxes(title='Accuracy')



    fig2.show()
