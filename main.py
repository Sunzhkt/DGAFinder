import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
import sys


def main(filepath):
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
    # 合并Alexa和DGA的数据
    train_data = alexa + dga
    # 特征提取，获得训练数据
    vectorizer = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\w', decode_error='ignore', max_features=1000)
    x_train = vectorizer.fit_transform(train_data)
    # 训练结果：alexa为0，dga为1
    y_train = ["notdga"]*len(alexa)+["dga"]*len(dga)

    # 读取测试文件，得到测试数据
    t = pd.read_csv(filepath[1], header=None)
    test_data = []
    for i in t.values:
        test_data.append(i[0])
    x_test = vectorizer.fit_transform(test_data)
    # 利用多层感知器算法（mlp）进行判断，得到判断结果
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(x_train, y_train)
    y_result = clf.predict(x_test)
    # 将判断结果输出到文件
    result = []
    for r in range(len(y_result)):
        result.append(test_data[r] + ", " + y_result[r] + '\n')
    output = open(filepath[2], 'w')
    output.writelines(result)


if __name__ == '__main__':
    main(sys.argv)
