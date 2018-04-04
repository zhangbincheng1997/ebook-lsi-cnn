# 智能阅读模型的构建
数据集下载：
1. http://www.tipdm.org/jingsa/1253.jhtml
2. https://spaces.ac.cn/archives/4338

## 构建流程
总体分为两部分：第一部分粗匹配相似句子，第二部分从候选答案中选择正确答案。

第一部分可使用 TF-IDF、LSI 等传统方法。

第二部分可使用 基于深度学习的问答系统。

本文主要构建第二部分，第一部分可参考：https://github.com/littleredhat1997/doc-similarity

## 目录结构

### word2vec 词向量
1. word2vec.ipynb
2. step.ipynb

### main 模型训练
1. data/data.ipynb
2. 1_FastText.ipynb
3. 2_CNN1.ipynb
4. 3_CNN2.ipynb
5. 4_BiLSTM.ipynb
6. 5_Attention.ipynb
7. evaluate.ipynb

### generalization 泛化测试
1. predict.ipynb
2. evaluate.ipynb

## 模型设计
1. FastText

![alt text](docs/FastText.png "title")

2. CNN

![alt text](docs/CNN.png "title")

3. Bi-LSTM

![alt text](docs/Bi-LSTM.png "title")

4. Attention

![alt text](docs/Attention.png "title")

### 数据预处理
生成一个词向量矩阵具体步骤如下所示：

```
①分词 (Cut)：
>>> ['第六届 泰迪杯 数据挖掘 挑战赛']

②字典化 (Tokenizer)：
>>> {'挑战赛': 4, '数据挖掘': 3, '泰迪杯': 2, '第六届': 1}

③序列化 (Sequences)：
>>> [[2, 3], [3, 4]]

④填充字符 (Padding)：
>>> array([[0, 0, 0, 0, 0, 0, 0, 0, 2, 3],
       [0, 0, 0, 0, 0, 0, 0, 0, 3, 4]], dtype=int32)
```

经过上面的过程 tokenizer保存了语料中出现过的词的编号映射。
