#简介#

这是我第一个比较完整的python程序，把‘‘UMDHMM‘‘的大部分功能用python重写了，没有经过完整的测试。
数据输入格式与UMDHMM基本相同，都是在后缀为‘‘.hmm‘‘的文件中包含初始向量‘‘pi‘‘、转移矩阵‘‘A‘‘、混淆矩阵
‘‘B‘‘，在后缀为‘‘.seq‘‘的文件中包含观察序列及其个数‘‘T‘‘。稍微不同的是，每个数据的后面比UMDHMM的输入数据
多一个逗号，你可以直接运行test_hmm.py观察结果。

[UMDHMM][1]是一款轻量级的HMM（Hidden Markov Model）C语言实现，更详细的说明可以参考
[《HMM学习最佳范例五：前向算法4》][2]。


##待做##

1. 添加注释及docstring
2. 测试viterbi算法
3. 代码优化

[1]: http://www.kanungo.com/software/software.html
[2]: http://www.52nlp.cn/hmm-learn-best-practices-five-forward-algorithm-4





