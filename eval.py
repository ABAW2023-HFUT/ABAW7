import numpy as np
from sklearn.metrics import f1_score

def CCC_score(y_pred, y_true):    
    print(y_pred.shape, y_true.shape)
    # print(np.mean((y_pred-y_true)**2))
    y_true_mean = np.mean(y_true, axis=0)
    y_pred_mean = np.mean(y_pred, axis=0)

    y_true_var = np.var(y_true, axis=0)
    y_pred_var = np.var(y_pred, axis=0)

    cov = np.mean((y_true - y_true_mean) * (y_pred - y_pred_mean), axis=0)
    
    ccc = np.mean(2.0 * cov / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2))
    ccc = np.squeeze(ccc)
    print(y_true_mean, y_pred_mean, y_true_var, y_pred_var, cov, ccc)
    return ccc

def VA_metric(x, y):
    return CCC_score(x, y)

def EXPR_metric(x, y):
    if not len(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            x = np.argmax(x, axis=-1)
    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)
    f1 = f1_score(x, y, average= 'macro')
    return f1

def averaged_f1_score(input, target):
    N, label_size = input.shape
    
    f1s = []
    for i in range(label_size):
        f1 = f1_score(input[:, i], target[:, i])
        f1s.append(f1)
    return np.mean(f1s)

def sigmod(x):
    return 1/(1+np.exp(-x))

def AU_metric(x, y):
    x = np.around(sigmod(x))
    f1_au = averaged_f1_score(x, y)
    return f1_au

if __name__ == '__main__':
    a = np.argmax(np.array([[0.1,0.1,0.9],[0.9,0.1,0.1],[0.1,0.9,0.1]]),axis=-1)
    b = np.array([2,0,1])
    print()
    print(f1_score(a, b, average= 'macro'))