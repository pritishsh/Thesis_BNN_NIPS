import numpy as np

def num_switches(old,new):
    change = (new - old) / 2
    change=change.astype(np.int64)
    num_n_to_p = change[change== 1].sum()
    num_p_to_n = change[change==-1].sum()*-1
    return num_n_to_p, num_p_to_n

def example_1():
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

def differences(a,b):
    #returns array with 1 for position where values are different, 0 otherwise
    c= (a-b)/(np.sign(a-b))/2
    c=np.nan_to_num(c,nan=0).astype(int)
    return c


if __name__ == '__main__':
    a = np.array([
        [1, 1, 1, 1, 1, - 1, 1, 1, 1],
        [1, 1, -1, 1, 1, 1, 1, 1, - 1],
        [1, 1, - 1, 1, 1, 1, 1, 1, 1]
    ])

    b = np.array([
        [1, -1, 1, 1, 1, 1, 1, 1, -1],
        [1, 1, 1, 1, 1, 1, 1, 1, - 1],
        [1, 1, - 1, 1, 1, 1, 1, - 1, -1]
    ])

    c = np.array([
        [ -1,1, 1, 1, 1, 1, 1, 1, -1],
        [ 1, 1,1, 1, 1, 1, 1, 1,- 1],
        [ 1, 1,- 1, 1, 1, 1, 1,- 1, -1]
    ])

    switchcount = np.zeros_like(a)
    print(switchcount)
    switchcount += differences(a,b)
    print(switchcount)
    switchcount += differences(b,c)
    print(switchcount)



