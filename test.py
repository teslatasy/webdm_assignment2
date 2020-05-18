#!/usr/bin/env python3
# coding: utf-8

import nlp
import pandas as pd
from snownlp import SnowNLP  # 情感分析语言处理库
from gensim import corpora, models
def str_unique(raw_str, reverse=False):
    """
    比如：我喜欢喜欢喜欢喜欢喜欢喜欢该商品；去掉重复的“喜欢”
    :param raw_str:
    :param reverse: 是否转置
    :return:
    """
    if reverse:
        raw_str = raw_str[::-1]
    res_str = ''
    for i in raw_str:
        if i not in res_str:
            res_str += i
    if reverse:
        res_str = res_str[::-1]
    return res_str

data_path = './comment.csv'	
df = pd.read_csv(data_path, encoding='gbk')
df = df.dropna()  # 消除缺失数据 NaN为缺失数据
df = pd.DataFrame(df.iloc[:, 0].unique())  # 去掉第一列的重复数据；iloc[:, 0]表示索引每一行的第一列；
ser1 = df.iloc[:, 0].apply(str_unique)	# 这时，因为索引了第一列，所以结果成了Series；
print('df2', type(ser1))  # <class 'pandas.core.series.Series'>
df2 = pd.DataFrame(ser1.apply(str_unique, reverse=True))	# 再次生成DataFrame；
df3 = df2[df2.iloc[:, 0].apply(len) >= 4]
# 语义积极的概率，越接近1情感表现越积极
coms = df3.iloc[:, 0].apply(lambda x: SnowNLP(x).sentiments)
print('情感分析后：')
positive_df = df3[coms >= 0.9]  # 特别喜欢的
negative_df = df3[coms < 0.1]  # 不喜欢的
print('特别喜欢的')
print(positive_df)
print('------------------')
print('不喜欢的')
print(negative_df)
nlp = nlp.NLP('CRF')
my_cut = lambda s: ' '.join(nlp.cut(s)) 
positive_ser = positive_df.iloc[:, 0].apply(my_cut)  # 通过“广播机制”分词，加快速度
negative_ser = negative_df.iloc[:, 0].apply(my_cut)
print('大于0.5---正面数据---分词')
print(positive_ser)
print('小于0.5---负面数据---分词')
print(negative_ser)
stop_list = './stoplist.txt'	 
stops = pd.read_csv(stop_list, encoding='gbk', header=None, sep='tipdm', engine='python')
stops = [' ', ''] + list(stops[0])  # pandas自动过滤了空格符，这里手动添加
positive_df = pd.DataFrame(positive_ser)
negative_df = pd.DataFrame(negative_ser)
positive_df[1] = positive_df[0].apply(lambda s: s.split(' '))  # 定义一个分割函数，然后用apply广播
positive_df[2] = positive_df[1].apply(lambda x: [i for i in x if i.encode('utf-8') not in stops])

negative_df[1] = negative_df[0].apply(lambda s: s.split(' '))  # 定义一个分割函数，然后用apply广播
negative_df[2] = negative_df[1].apply(lambda x: [i for i in x if i.encode('utf-8') not in stops])

print('去停用词后：positive_df')
print(positive_df)

print('------------------')
print('去停用词后：negative_df')
print(negative_df)

# 正面主题分析
pos_dict = corpora.Dictionary(positive_df[2])
pos_corpus = [pos_dict.doc2bow(i) for i in positive_df[2]]
pos_lda = models.LdaModel(pos_corpus, num_topics=3, id2word=pos_dict)
print('#正面主题分析')
for i in range(3):
    print('topic', i)
    print(pos_lda.print_topic(i))  # 输出每个主题

# 负面主题分析
neg_dict = corpora.Dictionary(negative_df[2])  # 建立词典

neg_corpus = [neg_dict.doc2bow(i) for i in negative_df[2]]  # 建立语料库

neg_lda = models.LdaModel(neg_corpus, num_topics=3, id2word=neg_dict)  # LDA 模型训练
print('#负面主题分析')
for i in range(3):
    print('topic', i)
    print(neg_lda.print_topic(i))  # 输出每个主题

# def test():
#     nlp = nlp.NLP('CRF')
#     text = '格力空调是真的很给力，赞！'

#     words = nlp.cut(text)
#     print('words', words)
#     postags = nlp.postag(words)
#     print('postags', postags)
#     ners = nlp.ner(text)
#     print('ners', ners)
#     deps = nlp.dep(words, postags)
#     print('deps',deps)
# test()
