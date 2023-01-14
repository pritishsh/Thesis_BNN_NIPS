import pandas as pd
import plotly as pt
import plotly.express as px

if __name__ == '__main__':
    #df = pd.DataFrame
    df=pd.read_excel('../500 cycles of LTP-LTD.xlsx', index_col=False)

    ltp_start =17378
    ltp_end =17429

    ltd_start=17428
    ltd_end=17489


    df_ltp = df.iloc[ltp_start:ltp_end,:]
    df_ltd = df.iloc[ltd_start:ltd_end,:]
    #print(df.iloc[:,0])
    #fig=px.scatter(x=df.iloc[:,0], y=df.iloc[:,1])
    #fig.show()

    df_ltd.to_csv('ltd.csv', header=False,index=False)
    df_ltp.to_csv('ltp.csv', header=False,index=False)


