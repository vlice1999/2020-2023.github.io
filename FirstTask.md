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
 |预测为假|False negative(FN)|True nagetive(NT)|
 
 #### 准确率与召回率
 Recall Rate    召回率/查全率 = TP/(TP + FN) 
 
 Precision Rate 准确率/查准率 = TP/(TP + FP)
 
 #### 准确度与F1-score
 Accuracy = $${TP + TN\over P + N}$$
 
 ### 我的疑问
 图片显示，归一化和扩展到[0,255]的原理比较不清楚。
 
 ## Day3
 
