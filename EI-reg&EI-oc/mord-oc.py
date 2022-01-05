#class mord.MulticlassLogistic(alpha=1.0, verbose=0, maxiter=10000)
import mord
import json
import numpy as np

def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = np.mean(x)
    my = np.mean(y)
    xm, ym = x - mx, y - my
    r_num = np.sum(xm * ym)
    x_square_sum = np.sum(xm * xm)
    y_square_sum = np.sum(ym * ym)
    r_den = np.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return np.mean(r)

# 载入数据
train = 'train-oc.json'
test = 'test-oc.json'
dev = 'dev-oc.json'

f1 = open(train, 'r')
train_data = json.load(f1)
f2 = open(test, 'r')
test_data = json.load(f2)
f3 = open(dev, 'r')
dev_data = json.load(f3)

X_test = np.array([xi['bert_vector'][0][:] for xi in test_data])  # x存储testsize*768的二维矩阵
y_test = np.array([float(yi['intensity']) for yi in test_data])  # y存储这些数据的intensity
X_train = np.array([xi['bert_vector'][0] for xi in train_data])  # x存储trainsize*768的二维矩阵
y_train = np.array([float(yi['intensity']) for yi in train_data])  # y存储这些数据的intensity
X_dev = np.array([xi['bert_vector'][0] for xi in dev_data])  # x存储trainsize*768的二维矩阵
Y_dev = np.array([float(yi['intensity']) for yi in dev_data])  # y存储这些数据的intensity




clf=mord.OrdinalRidge(alpha=70.0,max_iter=2000) #alpha:70 r: 75.76
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(f"pearson: {pearson_r(y_test,y_pred)}")

'''for i,res in enumerate(zip(y_pred,y_test)):
    if i==100:
        break
    pred,act=res
    print(f'pred:{pred} actual:{act}')'''