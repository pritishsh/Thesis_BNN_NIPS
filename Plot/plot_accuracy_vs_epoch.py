import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

def accuracy_plot(accuracy_data, averaged_samples=False, group_size=100, filetype="xlsx", skip_rows=0 ):
    '''
    accuracy_data is path to excel sheet containing columns: Accuracy, Epoch
    averaged_sample: pass true to create boxplots for each group of epochs
    group_size: how many samples for each boxplot
    '''

    if filetype == "xlsx":
        df = pd.read_excel(accuracy_data)

    if filetype=="csv":
        df = pd.read_csv(accuracy_data)

    #df = df.drop(axis=0,index=range(1,skip_rows))
    #df = df.reset_index(drop=True)
    epoch_series, accuracy_series = df['epoch'], df['accuracy']
    trace_colors = ['lightsalmon', 'orange', 'darkcyan']
    fig2 = make_subplots(rows=1,cols=1)

    # classic plot with each datapoint displayed on graph
    if averaged_samples == False:
        fig2.add_trace(go.Scatter(
            name='Accuracy',
            x=epoch_series,
            y=accuracy_series,
            #mode='lines+markers',
            mode='markers+lines',
            opacity=1,
            line = dict(color = trace_colors[1], width=1),
            marker = dict(color=trace_colors[1], size=5),
        ),
            row=1,
            col=1
        )

        fig2.update_layout(
            paper_bgcolor='white',
            showlegend=True,
            legend_orientation='h',
            legend = dict(
                bgcolor='white',
                bordercolor='black',
                xanchor='right',
                x=0.95,
                yanchor='bottom',
                y=0.05,
            ),
            font = dict(
                family='Times New Roman',
                size=20,
            ),
            title = {
                'text': 'Accuracy v/s Epoch',
                'x': 0.5
            }
        )

        fig2.update_xaxes(title='Epoch')
        fig2.update_yaxes(title='Accuracy (%)')

    # Plot box plots for every group of epochs
    if averaged_samples == True:

        # group column specifies x-axis value for individual box plot
        df['group'] = 'none'
        for i in range(0, int(len(df)/group_size) +1 ):

            #naming of each group
            start_loc, end_loc = i*group_size, min( (i+1)*group_size, len(df) )
            boxplot_df = df.iloc[ start_loc : end_loc ]
            boxplot_df['group'] = str(start_loc) + ' to ' + str(end_loc)
            df.update(boxplot_df)

        fig2 = px.box(df, x='group', y='accuracy', points= False)
        # Just changes the x-axis name from 'group' to Epoch
        fig2.update_xaxes({'title':'Epoch'})
        fig2.update_layout({'title' : 'Accuracy vs Epoch'})
    fig2.show()


if __name__=='__main__':
    #results_path = '../saved_models/8000_epochs.xlsx'
    #results_path = '../saved_models/74242_epochs.xlsx'
    #results_path = '../Neurosim/data/VGG-8 CIFAR10 5bit noD2D onoff=10 500 epochs/output_accuracy.xlsx'
    results_path = '../Neurosim/data/VGG-8 CIFAR10 with_D2D 6-bit/20 March from Amit/PythonWrapper_Output_modified.csv'
    #group_labels = ['Accuracy', '1 to -1']
    #accuracy_plot(results_path )
    accuracy_plot(results_path, averaged_samples=False, filetype="csv", skip_rows=420)
