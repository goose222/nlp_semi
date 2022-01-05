import numpy as np
from tensorflow.python.keras.models import load_model

np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense, Dropout
import json
from keras import backend as K
from keras import regularizers
from keras.models import model_from_json
import matplotlib.pyplot as plt

# 自定义目标函数：pearson相关系数
def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

# 载入数据
train = 'train-reg.json'
test = 'test-reg.json'
dev = 'dev-reg.json'

f1 = open(train, 'r')
train_data = json.load(f1)
f2 = open(test, 'r')
test_data = json.load(f2)
f3 = open(dev, 'r')
dev_data = json.load(f3)

X_test = np.array([xi['bert_vector'][0][:] for xi in test_data])  # x存储testsize*768的二维矩阵
Y_test = np.array([float(yi['intensity']) for yi in test_data])  # y存储这些数据的intensity
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

model = Sequential()
# relu能达到0.71
# tanh能达到0.694
model.add(Dense(input_dim=768, units=80, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
print(model.summary())
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', pearson_r])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1000, batch_size=1701)
# model.fit(X, Y, validation_split=0.2, epochs=900, batch_size=2400)

# 保存模型和参数
model.save('model.h5')
model.save_weights('my_model_weights.h5')

# 准确率
scores = model.evaluate(X_test, Y_test, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[2], scores[2] * 100))

# 计算dev上的误差
predicted = model.predict(X_test)
result = np.mean(abs(predicted - Y_test))
print("The mean error of linear regression:", result)
plt.figure(num=2, figsize=(5, 5), dpi=100)
plt.scatter(predicted, Y_test)
plt.show()

# 把预测值存入tsv文件
f2 = open("EI-reg-pred-label.tsv", "w", encoding='utf-8')
text2 = 'idx\tpred_label\n'
f2.write(text2)
i = 0
for label in predicted:
        i = i+1
        f2.write(str(i) + '\t' + str(label[0]) + '\n')
f2.close()



# 载入之前训练的模型，用于evaluate
'''model = load_model("model.h5", custom_objects={'pearson_r': pearson_r})
model.load_weights('my_model_weights.h5')
scores = model.evaluate(X_dev,Y_dev,verbose=0)
# 把metrics_names[2]和scores[2]改成metrics_names[1], scores[1]，输出的是mae的值
print('%s: %.2f%%' % (model.metrics_names[2], scores[2]*100))'''