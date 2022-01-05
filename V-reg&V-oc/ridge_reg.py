import numpy as np
from tensorflow.python.keras.models import load_model
import json
np.random.seed(1337)

from sklearn.linear_model import Ridge

# 自定义目标函数：pearson相关系数
def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = np.mean(x, axis=0)
    my = np.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = np.sum(xm * ym)
    x_square_sum = np.sum(xm * xm)
    y_square_sum = np.sum(ym * ym)
    r_den = np.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return np.mean(r)

# 载入数据
train = 'V-data-vector/2018-Valence-reg-En-train_new-vector.json'
test = 'V-data-vector/2018-Valence-reg-En-test-gold_new-vector.json'
dev = 'V-data-vector/2018-Valence-reg-En-dev_new-vector.json'

f1 = open(train, 'r')
train_data = json.load(f1)
f2 = open(test, 'r')
test_data = json.load(f2)
f3 = open(dev, 'r')
dev_data = json.load(f3)

X_test = np.array([xi['bert_vector'][0][:] for xi in test_data])  # x存储testsize*768的二维矩阵
Y_test = np.array([float(yi['intensity'])  for yi in test_data])  # y存储这些数据的intensity
X_train = np.array([xi['bert_vector'][0] for xi in train_data])  # x存储trainsize*768的二维矩阵
Y_train = np.array([float(yi['intensity']) for yi in train_data])  # y存储这些数据的intensity
X_dev = np.array([xi['bert_vector'][0] for xi in dev_data])  # x存储trainsize*768的二维矩阵
Y_dev = np.array([float(yi['intensity']) for yi in dev_data])  # y存储这些数据的intensity



# 增加一维
'''X_train = np.expand_dims(X_train.astype(float), axis=2)
Y_train = np.expand_dims(Y_train.astype(float), axis=1)
X_test = np.expand_dims(X_test.astype(float), axis=2)
Y_test = np.expand_dims(Y_test.astype(float), axis=1)
X_dev = np.expand_dims(X_dev.astype(float), axis=2)
Y_dev = np.expand_dims(Y_dev.astype(float), axis=1)'''

# 把train和test合并起来，之后做交叉验证
'''X = np.vstack((X_train, X_test))
Y = np.hstack((Y_train, Y_test))'''

model=Ridge(alpha=100.0)
model.fit(X_train,Y_train)



print("###########################################start eval###########################################")
# 准确率
predicted = model.predict(X_test)

print(f"pearson: {pearson_r(Y_test,predicted)}")



f2 = open("V-reg-pred-label-2.tsv", "w", encoding='utf-8')
text2 = 'idx\tpred_label\n'
f2.write(text2)
i = 0
for label in predicted:
        i = i+1
        f2.write(str(i) + '\t' + str(label) + '\n')
f2.close()


# 载入之前训练的模型，用于evaluate
'''
model = load_model("model.h5", custom_objects={'pearson_r': pearson_r})
model.load_weights('my_model_weights.h5')
scores = model.evaluate(X_dev,Y_dev,verbose=0)
# 把metrics_names[2]和scores[2]改成metrics_names[1], scores[1]，输出的是mae的值
print('%s: %.2f%%' % (model.metrics_names[2], scores[2]*100))
'''

#73.39 a=10
#75.54 a=20
#76.37 a=30
#76.86 a=40
#77.36 a=60
#77.66 a=100