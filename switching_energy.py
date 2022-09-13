import numpy as np

def num_switches(old,new):
    change = (new - old) / 2
    change=change.astype(np.int64)
    num_n_to_p = change[change== 1].sum()
    num_p_to_n = change[change==-1].sum()*-1
    return num_n_to_p, num_p_to_n



if __name__ == '__main__':

    a = np.array([
        [ 1, 1, 1, 1, 1,- 1, 1, 1, 1],
        [ 1, 1,-1, 1, 1, 1, 1, 1,- 1],
        [ 1, 1,- 1, 1, 1, 1, 1, 1, 1]
    ])

    b = np.array([
        [ 1,-1, 1, 1, 1, 1, 1, 1, -1],
        [ 1, 1,1, 1, 1, 1, 1, 1,- 1],
        [ 1, 1,- 1, 1, 1, 1, 1,- 1, -1]
    ])

    print(num_switches(a,b))
    x,y = np.array([0,0])
    x,y = np.array([x,y]) + num_switches(a,b)
    x,y = np.array([x,y]) + num_switches(a,b)
    print(x,y)


