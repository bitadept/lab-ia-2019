# a.
y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
y_pred = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]

def acc(yr, yp):
    s = 0
    for idx in range(len(yr)):
        if(yr[idx] == yp[idx]):
            s += 1
    return s / len(yr)

print(acc(y_true, y_pred))

# b.
# true positive, false positive, false negative
def precision_recall_score(yr, yp):
    tp = fp = fn = 0

    for i in range(len(yr)):
        if yr[i] == yp[i] == 1:
            tp += 1
        elif yr[i] == 1 and yp[i] == 0:
            fn += 1
        elif yr[i] == 0 and yp[i] == 1:
            fp += 1
    return tp / (tp + fn), tp / (tp + fp)

print(f'{precision_recall_score(y_true, y_pred)}')
