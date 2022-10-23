import numpy as np
import pandas as pd

pq = {
    'hedcc': [4,7,2,8,1,8],
    '34dcc': [4,2,2,8,2,3]
}

df = pd.DataFrame(pq)

path = 'saved_models/1309/try/'
df.to_excel(path+'try.xlsx')

