# 第一次任务

## Day0
### 任务内容：
阅读自动驾驶安全相关的文献，两周内写一个总结报告
跑自动驾驶程序的demo

### 课程目标：
自动驾驶目标的欺骗与对抗：比如，车牌号的检测逃逸/路标的检测逃逸。
通俗的说，本来车牌号能被摄像头的识别程序识别出来，通过欺骗方 法，让车牌号检测不出来；本来红绿灯能检测出来，欺骗以后检测不出来；本来路上的线提示要右拐了，欺骗以后就识别不出拐弯线。

### 推荐书籍：
 ·《机器学习与安全》
 
 
 ·《AI安全与对抗样本入门》 
 
 
 ·《网络空间欺骗》 
 
 
 ·《动态目标防御》
 
 ### 我的疑问：
 欺骗是什么，怎么做到的，可以用哪些方法，如何检验
 
 ## Day1
 ### 自动驾驶汽车的欺骗：
 通过一些措施使自动驾驶程序无法正常运行，比如识别错限速车标[1]，误读路况中的障碍物导致停车[2]，
 错误的认为绿灯是红灯导致自动驾驶汽车停止[2]等等。具体措施：给路标贴电子胶带，在路上放平面广告等等。

 ### 深度学习的脆弱性：
 1.偷取模型 2.数据投毒 3.对抗样本（白盒攻击，黑盒攻击，真实世界/物理攻击）
 ### 常见检测和加固方法：
 检测过程：即为发起攻击的过程 
 #### 基于白盒：
 ILCM（最相似替代算法）
 
 FGSM（快速梯度算法）
 
 BIM（基础迭代算法）
 
 JSMA（显著图攻击算法）
 
 DeepFool（DeepFool算法）
 
 C/W（C/W算法）
 #### 黑盒攻击方法：
 Single Pixel Attack（单像素攻击）
 
 Local Search Attack（本地搜索攻击）
 ### 深度学习脆弱性加固
 · Feature squeezing(特征凝结)
 
 · Spatial smoothing（空间平滑）

 · Label smoothing(标签平滑)
 
 · Adversarial training（对抗训练）
 
 · Virtual adversarial training（虚拟对抗训练）
 
 · Gaussian data augmentation（高斯数据增强）

 ### 参考文献：
 [1]: 新浪科技 《研究人员骗过特斯拉汽车：把35英里限速看成85英里》 url:http://www.techweb.com.cn/internet/2020-02-20/2777981.shtml
 
 
 [2]:生物学谢博士 《自动驾驶汽车感知曝漏洞：很容易被假路标欺骗》url:https://baijiahao.baidu.com/s?id=1660646822297982069&wfr=spider&for=pc
 
 ## Day2
 阅读《AI安全之对抗样本入门》
 ### 深度学习基本过程
 #### 1.数据预处理
 #### 2.定义网络结构
 Dense层，Activation层，DropOut层，Embedding层（把词映射到一个维度的向量中），Flatten层（将输入压平，把多维的数据一维化），Permute层（按照深度学习框架指定的模式重排），Reshape层（将输入的shape转换成特定的shape）
 #### 3.定义损失函数
 #### 4.反向传播和优化器
 优化器一般包括SGD，RMSprop，Adam
 #### 5.范数
 · L0范数：用于度量非零元素的个数，在对抗样本中，指对抗样本相对原始图片，改变像素的个数
 
 · L1范数：曼哈顿距离，最小绝对误差等。使用L1范数可以度量两个向量间的差异，表示向量元素中非零元素的绝对值之和。
 
 · L2范数：向量元素的平方再开方。再对抗样本中，通常指对抗样本相对原始图片，所修改像素的变化量的平方和再开方。
 
 · 无穷范数：L<sub>inf</sub>，用于度量向量元素的最大值。在对抗样本中L<sub>inf</sub>范数通常指的是对抗样本相对原始图片，所修改像素的变化量绝对值的最大值。
 ### 基于CNN的图像分类
 #### CNN的常见组件
 卷积层，池化层，全连接层，非线性变化层，Dropout
 #### VGG的结构
 ![vgg16](https://github.com/vlice1999/2020-2023.github.io/blob/master/vgg_layers.png)
 #### 可视化CNN
 以VGG16为例，源代码路径：https://github.com/duoergun0729/adversarial_examples/blob/master/code/1-case2-keras.ipynb
 ```py
 # 使用vgg16
 from keras.applications.vgg16 import VGG16
 import matplotlib.pyplot as plt
 %matplotlib inline
 # 加载基于ImageNet2012数据集预训练的参数
 model = VGG16(weights = 'imagenet')
 
 # 加载图片（Pig）
 from keras.preprocessing import image
 from keras.applications.vgg16 import preprocess_input, decode_predictions
 import numpy as np
 
 # 图片（Pig）的路径
 img_path = input("请输入图片的路径：")
 # 调整图片到指定大小 224*224
 img = image.load_img(img_path,target_size = (224,224))
 # 展示图片
 plt.imshow(img)
 plt.show()
 # 扩展维度，适配模型输入大小(1,224,224,3)
 x = np.expend_dims(x,axis = 0);
 # 图像预处理
 x = preprocess_input(x)
 
 # 对图片进行预测
 preds = model.predict(x)                                    #得到预测结果
 print('Predicted:', decode_predictions(preds, top = 3)[0])  #展示前三的预测
 
 from keras import models
 layer_names = ['block1_conv1', 'block3_conv1', 'block5_conv1']
 # 获取指定层的输出:
 layer_outputs = [model.get_layer(layer_name).output for layer_name in layer_names]
 # 创建新的模型，该模型的输出为指定的层的输出
 activation_model = models.Model(inputs = model.input, outputs = layer_outputs)
 
 # 获得图片（pig）的输出
 activations = activation_model.predict(x)
 
 images_per_row = 8
 
 for layer_name, layer_activation in zip(layer_names, activations):
  # 获取卷积核个数
  n_features = layer_activation.shape[-1]
  # 特征图的形状（1,size,size,n_features）
  size = layer_activation.shape[1]
  # 最多展示八行
  n_cols = 8
  display_grid = np.zeros((size * n_cols, images_per_row * size)) #画板
  for col in range(n_cols):
   for row in range(images_per_row):
    channel_image = layer_activation[0,:,:,col*images_per_row + row] #范围不固定，需要根据均值和方差进行归一化
    channel_image -= channel_image.mean()
    channel_image /= channel_image.std()
    # 数据扩展到[0,255]范围
    channel_image *= 128
    channel_image += 128
    channel_image = np.clip(channel_image,0,255).astype('uint8')
    display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
  # 展示图片
  scale = 1. / size
  plt.figure(figsize = (scale * display_grid.shape[1], scale * display_grid.shape[0]))
  plt.title(layer_name)
  plt.grid(False)
  plt.imshow(display_grid,aspect = 'auto',cmap = 'viridis')
 plt.show()
 ```
 ### 常见衡量指标
 #### 测试数据
 #### 混淆矩阵
 
 |        |实际为真|实际为假|
 |--------|:--------:|:-------:|
 |预测为真|True positive(TP)|False positive(FP)|
 |预测为假|False negative(FN)|True nagetive(TN)|
 
 #### 准确率与召回率
 Recall Rate    召回率/查全率 = TP/(TP + FN) 
 
 Precision Rate 准确率/查准率 = TP/(TP + FP)
 
 #### 准确度与F1-score
 
 $Accuracy$ = $\frac {TP + TN}{P + N}$ = $\frac {TP + TN}{TP + TN + FN + FN}$

 $F1-Score$ = $\frac {2TP}{2TP + FP + FN}$
 #### ROC与AUC
 ROC(Recevier Operating Characteristic，受试者工作特征)曲线。以真阳性率为纵坐标，假阳性率为横坐标绘制的曲线。
 
 真阳性率 TPR = TP/(TP + FN)
 
 假阳性率 FPR = FP/(FP + TN)
 
 曲线越接近左上角，说明分类效果越好。因为（0，1）代表完美分类。
 
 AUC(Area Under the Recevier Operating Characteristic)曲线是量化衡量ROC分类性能的指标，指ROC曲线所覆盖的面积，越大（越接近1）越好。
 
 ### 集成学习（Ensemble Learning）
 使用一系列学习器进行学习，并使用某种规则把各个学习结果进行整合从而获得比单个学习器更好的学习效果的一种机器学习方法。如果使用的多个分类器相同，则为同质；否则为异质。综合判断的策略分为两种：加权平均和投票。
 可以粗略的分为两类，第一类是个体学习器之间存在强依赖关系，一系列个体学习器基本都需要串行生成，代表算法为Boosting系列算法。第二类是个体学习器不存在依赖关系，一系列个体学习器可以并行生成，代表算法是Bagging算法和随机森林（Random Forest）系列算法。
 #### Boosting算法（待补全）
 #### Bagging算法（待补全）
 
 ### 我的疑问
 图片显示，归一化和扩展到[0,255]的原理比较不清楚。
 
 ## Day3
 ### 今日目标
 ① TensorFlow，Pytorch入门，学习简单网络的搭建
 
 ② 图像处理算法回顾
 
 ③ 白盒攻击算法入门
 ### Tensorflow 
 源代码：https://github.com/duoergun0729/adversarial_examples/blob/master/code/2-tensorflow.ipynb
 #### 1.加载相关库
 ```py
 import tensorflow as tf
 from tensorflow.examples.tutorials.mnist import input_data
 from tensorflow.python.framework import graph_util
 import os
 ```
 #### 2.加载数据集
 mnist数据集,链接 http://yann.lecun.com/exdb.mnist/
 
 默认形状为[28,28,1]，为了便于处理，改变其维度为一维向量784。
 
 特征数据归一化：默认为[0,255]，通过除以255完成
 
 默认的标签数据类型为整数，取值范围为0到9。为了便于网络训练，把标签数据转换为One-Hot编码（独热编码）。
 
 |文件名称|文件用途|
 |:-----:|:------:|
 |train-images-idx3-ubyte.gz|60000个图片训练样本|
 |train-labels-idx1-ubyte.gz|60000个图片训练样本的标注|
 |t10k-images-idx3-ubyte.gz|10000个图片测试样本|
 |t10k-labels-idx1-ubyte.gz|10000个图片训练样本的标注|
 
 ```py
 mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
 ```
 #### 3.定义网络结构
 输入层大小784，隐含层300个节点，激活函数relu，中间为了避免过拟合，使用DropOut层，输出层大小为10，激活函数为softmax。为了导出导入pb文件方便，将输入命名为input，将输出命名为output。
 
 ```py
 in_units = 784 # 输入节点个数
 h1_units = 300 # 隐藏层节点数
 
 # 初始化隐藏层权重W1，服从默认设置为0，标准差为0.1的截断正态分布
 W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev = 0.1))
 
 b1 = tf.Variable(tf.zeros([h1_units])) #隐含层偏置b1全部初始化为0
 W2 = tf.Variable(tf.zeros([h1_units,10])) #输出层权重初始化
 b2 = tf.Variable(tf.zeros[10]) #输出层
 x = tf.placeholder(tf.float32,[None, in_units], name = "input")
 keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
 
 # 定义模型结构
 hidden1 = tf.nn.relu(tf.matmul(x,W1) + b1)
 hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
 y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2, name = "output")
 ```
 #### 4.定义损失函数和优化器
 本例为多分类问题，采用交叉熵定义损失函数，使用Adagrad优化器
 ```py
 # 定义损失函数和优化器
 y_ = tf.placeholder(tf.float32, [None, 10])
 cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices = [1]))
 train_step = tf.train.AdagradOptimizer(0.3).minisize(cross_entropy)
 ```
 #### 5.训练与验证
 初始化参数，迭代训练5000轮，每轮训练的批量大小为100，为了抵御过拟合，每轮训练时仅通过75%的数据。每训练200批次，打印中间结果。
 ```py
 sess.run(tf.global_variables_initializer())
 correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_, 1))
 accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 for i in range(0,5000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  _, loss = sess.run([train_step, cross_entropy],{x: batch_xs, y:batch_ys, keep_prob:0.75})
  
  if i%200 == 0:
   acc = accuracy.eval(feed_dict = {x:mnist.test.images, y_:mnist.test.labels, keep_prob:1})
   print("loss = {}, acc = {}".format(loss, acc))
 ```
 
 ### 我的疑问
 1. 不会tf.palceholder的使用
 
 2. 损失函数，优化器的使用。
 
 3. tensorflow保存模型文件
