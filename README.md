# lambdaFM
## 前言：
* lambdaFM是lambdaRank和FM(Factorization Machines)的结合，用于解决排序问题。实现了pairwise和lambdaRank两种训练方法，可通过-rank参数选择。
lambdaFM是在[alphaFM](https://github.com/CastellanZhang/alphaFM)的代码基础上修改而成，同样是单机多线程版本，同样是FTR优化算法。<br>

* lambdaFM和alphaFM类似，同样适用于真实业务中大规模数据、高维稀疏特征的训练。由于采用FTRL优化算法，样本只需过一遍，不占用内存。通过管道的方式接受输入。<br>

比如训练样本存储在hdfs上，一个典型的使用方法是这样：<br>
训练：10个线程计算，factorization的维度是8，最后得到模型文件fm_model.txt<br>
`hadoop fs -cat train_data_hdfs_path | ./lambdafm_train -core 10 -dim 1,8 -rank ndcg -m fm_model.txt`<br>
测试：10个线程计算，factorization的维度是8，加载模型文件fm_model.txt，最后输出预测结果文件fm_pre.txt<br>
`hadoop fs -cat test_data_hdfs_path | ./lambdafm_predict -core 10 -dim 8 -m fm_model.txt -out fm_pre.txt`<br>

* lambdaFM同样支持加载上次模型继续训练，以及通过-fvs参数加强v的稀疏性。<br>

* 注意：dim参数有些变化，不再需要指定偏置项，所以只有<k1,k2>两项。当将dim参数设置为1,0时，lambdaFM就退化成lambdaLR。<br>

## 安装方法：
直接在根目录make即可，编译后会在bin目录下生成两个可执行文件。如果编译失败，请升级gcc版本。
## 输入文件格式：
类似于RankLib和SVMrank的格式，但更加灵活：特征编号不局限于整数也可以是字符串；特征值可以是整数或浮点数（特征值最好做归一化处理，否则可能会导致结果为nan），
特征值为0的项可以省略不写；qid可以是数字也可以是字符串；label（即相关性分数）必须是0,1,2,3等非负整数；#后面是注释，注释也可为空。
相同qid的数据必须相邻在一起，且按照自然展现的顺序排列。举例如下：<br>
`1 qid:1 sex:1 age:0.3 f1:1 f3:0.9 # 1AAAAA`<br>
`0 qid:1 sex:0 age:0.7 f2:0.4 f5:0.8 f8:1 # 1BBBBB`<br>
`3 qid:1 sex:1 age:0.3 f1:1 f3:0.9 # 1CCCCC`<br>
`2 qid:ab sex:0 age:0.2 f2:0.2 f8:1`<br>
`1 qid:ab sex:1 age:0.5 f1:1 f3:0.3`<br>
`4 qid:ab sex:0 age:0.1 f2:0.7 f5:0.2 f8:1`<br>
`...`<br>
## 模型文件格式：
`feature_name w v1 v2 ... vf w_n w_z v_n1 v_n2 ... v_nf v_z1 v_z2 ... v_zf`
## 预测结果格式：
`label qid score`<br>

## 参数说明：
### lambdafm_train的参数：
-m \<model_path\>: 设置模型文件的输出路径。<br>
-dim \<k1,k2\>: k1为1表示使用w参数，为0表示不使用；k2为v的维度，可以是0。	default:1,8<br>
-init_stdev \<stdev\>: v的初始化使用均值为0的高斯分布，stdev为标准差。	default:0.1<br>
-w_alpha \<w_alpha\>: w0和w的FTRL超参数alpha。	default:0.05<br>
-w_beta \<w_beta\>: w0和w的FTRL超参数beta。	default:1.0<br>
-w_l1 \<w_L1_reg\>: w0和w的L1正则。	default:0.1<br>
-w_l2 \<w_L2_reg\>: w0和w的L2正则。	default:5.0<br>
-v_alpha \<v_alpha\>: v的FTRL超参数alpha。	default:0.05<br>
-v_beta \<v_beta\>: v的FTRL超参数beta。	default:1.0<br>
-v_l1 \<v_L1_reg\>: v的L1正则。	default:0.1<br>
-v_l2 \<v_L2_reg\>: v的L2正则。	default:5.0<br>
-core \<threads_num\>: 计算线程数。	default:1<br>
-im \<initial_model_path\>: 上次模型的路径，用于初始化模型参数。如果是第一次训练则不用设置此参数。<br>
-fvs \<force_v_sparse\>: 为了获得更好的稀疏解。当fvs值为1, 则训练中每当wi = 0，即令vi = 0；当fvs为0时关闭此功能。	default:0<br>
-rank \<ranking_loss\>: pairwise或ndcg，支持两种排序算法，ndcg即lambdaRank算法。	default:pairwise<br>
-fast_mode \<fast_mode\>: 提供了一种快速训练模式，在v较大时速度能有一定提升，但效果可能会变差，谨慎使用。1表示开启，0表示关闭。	default:0<br>
### lambdafm_predict的参数：
-m \<model_path\>: 模型文件路径。<br>
-dim \<factor_num\>: v的维度。	default:8<br>
-core \<threads_num\>: 计算线程数。	default:1<br>
-out \<predict_path\>: 输出文件路径。<br>


