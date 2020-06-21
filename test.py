import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn import metrics

# 读Alexa数据
a = pd.read_csv("top-1m.csv", sep=",", header=None)
alexa = []
for i in a.values:
    alexa.append(i[1])
# 读DGA数据
d = pd.read_csv("dga.txt", sep="\t", header=None, skiprows=18)
dga = []
for i in d.values:
    dga.append(i[1])
# 合并数据
data = alexa + dga
# 特征提取
vectorizer = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\w', decode_error='ignore', max_features=1000)
x = vectorizer.fit_transform(data)
# 训练结果：alexa为0，dga为1
y = [0]*len(alexa)+[1]*len(dga)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# print(y_pred)
# print(classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
