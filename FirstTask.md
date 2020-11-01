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
 ### 张量（tensor）和计算图/数据流图
 以Tensorflow为例：
 ```py
 import tensorflow as tf
 
 a = tf.placeholder(tf.int64)
 b = tf.placeholder(tf.int64)
 c = tf.placeholder(tf.int64)
 
 d = (a + b) * c
 
 with tf.Session() as sess:
  print(sess.run(d, feed_dict = {a:1,b:2,c:3}))
 ```
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
 
 ### PyTorch入门
 源代码：https://github.com/duoergun0729/adversarial_examples/blob/master/code/2-pytorch.ipynb
 
 #### 1.加载相关库
 ```py
 import os
 import torch
 import torchvision
 from torch.autograd import Variable
 import torch.utills.data.dataloader as Data
 ```
 #### 2.加载数据集
 PyTorch对常见的数据集进行了封装。PyTorch对每个Tensor包括输入节点，并且都可以有自己的梯度值，因此训练数据要设置train = True，测试数据集要设置为train = False
 ```py
 train_data = torchvision.datasets.MNIST('dataset/mnist-pytorch',train = True, transform =  torchvision.transforms.ToTensor(), download = True)
 test_data = torvhvision.datasets.MNIST('dataset/mnist-pytorch',train = False, transform = torchvision.transforms.ToTensor())
 # 数据归一化
 # transform = tranforms.Compose([torchvision.transforms.Totensor(),torchvision.transforms.Normalize([0.5],[0.5] )])
 ```
 #### 3.定义网络结构
 使用两层网络结构，使用BatchNorm层替换Dropout层，在抵御过拟合的同时加快了训练的收敛速度。在PyTorch中定义网络结构，通常需要继承torch.nn.Module类。
 
 在forward中完成前向传播的定义，在init中完成主要网络层的定义：
 ```py
 class Net(torch.nn.Moudle):
  def __init__(self):
   super(Net, self).__init__()
   self.dense = torch.nn.Sequential(
    # 全连接层
    torch.nn.Linear(784,512),
    # BatchNorm层
    torch.nn.BatchNormld(512),
    torch.nn.ReLU(),
    torch.nn.Linear(512,10),
    torch.nn.ReLU()
   )
   
  def forward(self, x):
   x = x.view(-1,784)
   x = self.dense(x)
   return torch.nn.functional.log_softmax(x, dim = 1)
 ```
 #### 4.定义损失函数和优化器
 损失函数使用交叉熵CrossEntropyLoss，优化器使用Adam，优化的对象是全部参数：
 ```py
 optimizer = torch.optim.Adam(model.parameters())
 loss_func = torch.nn.CrossEntropyLoss()
 ```
 #### 5.训练与验证
 训练阶段把训练数据进行前向传播后，使用损失函数计算训练数据的真实标签与预测标签的损失值，然后显示调用反向传递backword()，使用优化器来调整参数。
 ```py
 for i, data in enumerate(train_loader):
  inputs, labels = data
  inputs, labels = inputs.to(device), labels.to(device)
  
  # 梯度清零
  optimizr.zero_grad()
  # 前向传播
  outputs = model(inputs)
  loss = loss_func(outputs, labels)
  # 反向传递
  loss.backward()
  optimizer.step()
 ```
 训练过程可视化，打印训练的中间结果，例如每100个批次打印下平均损失值。
 ```py
 sum_loss += loss.item()
 if (i + 1) % 100 == 0:
  print('epoch = %d, batch = %d loss: %0.4f' %(eporch+1, i+1, sum_loss/100))
  sum_loss = 0.0
 ```
 验证阶段，手动关闭反向传递，通过torch.no_grad()实现：
 ```py
 with totch.no_grad():
  correct = 0
  total = 0
  for data in test_loader:
   images, labels = data
   images, labels = images.to(device), labels.to(device)
   outputs = model(images)
   # 取得分最高的那个类
   _, predicted = torch.max(outputs.data, 1)
   total += labels.size(0)
   correct += (predicted == labels).sum()
   print('epoch = %d accuracy = %0.2f%%' % (epoch + 1,(100 * correct / total)))
 ```
保存模型文件为pth:
```py
torch.save(model.state_dict(), 'model/pytorch-mnist.pth')
```
### 使用预训练模型
#### 使用PyTorch进行图片分类
PyTorch通过torchvision库封装了对预训练模型的下载和使用。模型的预训练权重将下载到~/.torch/models/并在载入模型时自动载入。

加载需要的Python库，并对图像进行预处理。使用基于Imagenet数据集训练的ResNet50模型，图片大小转换成（224，224）。PyTorch在处理图片时，信道放在第一个维度，所以实际输入图像为（3，224，224）.

PyTorch加载预训练模型后，默认为训练模式，需要手动调成eval（预测）模式。
```py
import os
import numpy as np
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
# 手工调用eval方法进入预测模式
resnet50 = models.resnet50(pretrained = True).eval()
path = input("Image path:")
img = Image.open(path)
img = img.resize(224,224)
img = np.array(img).copy().astype(np.float32)
```
手工标准化处理图片
```py
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.255]
img /= 255.0
img = (img - mean)/std
img = img.transpose(2,0,1)
img = np.expend_dims(img, axis = 0)
img = Variable(torch.from_numpy(img).float())
```
对图片进行预测
```py
label = np.argmax(resnet50(img).data.cpu().numpy())
print("label = {}".format(label))
```
标签与物体对应关系参考链接：

https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
#### 使用Tensorflow进行图片分类
Tensorflow的模型多以pb文件保存，以Inception为例，模型文件为pb格式，其中的classify_image_graph_def.pb文件就是训练好的Inception模型，imagenet_synset_to_human_label_map.txt是类别文件

```
wegt http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
tar  -zxvf inception-2015-12-05.tgz
x classify_image_graph_def.pb
x cropped_panda.jpg
x imagenet_2012_challenge_label_map_proto.pbtxt
x imagenet_synset_to_human_label_map.txt
x LICENSE
```
图片数据的预处理在Inception的计算图中完成
```py
path = "../picture/pig.jpg"
image_data = tf.gfile.FastGFile(path, "rb").read()
```
加载pb文件，在会话中还原完整的计算图以及网络中的各层参数。
```py
session = tf.Session()
def create_graph(dirname):
 with tf.gfile.FastGFile(dirname, 'rb') as f:
  graph_def = session.graph_def
  graph_def.PaserFromString(f.read())
  _ = tf.import_graph_def(graph_def, name = )
ceate_graph("models/classify_image_graph_def.pb")
session.run(tf.global_varibales_initializer())
```
获取输入节点和输出节点，运行计算图获得结果
```py
logits = session.graph.get_tensor_by_name('softmax/logits:0')
x = session.graph.get_tensor_by_name('DecodeJpeg/contents:0')
predictions = session.run(logits, feed_dict = {x:image_data})
predictions = np.squeeze(predictions)
top_k = predicitons.argsort()[-3:][::-1]
for node_id in top_k:
 human_string = node_lookup.id_to_string(node_id)
 score = predictions[node_id]
 print('%s (score = %.5f)' % (human_string, score))
```
把pb文件的结构打印出来
```py
tensorlist =[n.name for n in session.graph_def.node]
print(tensorlist)
```
### 书P74-P75（未完）

### 我的疑问
 1. Tensorflow 损失函数，优化器的使用。
 
 2. tensorflow保存模型文件
 
 ## Day4
 我有了使自己震惊到的一个想法，在训练神经网络时加入图像的频谱图会不会增加图像识别的准确性。
 ### 图像格式
 以opencv为例，源代码链接:http://github.com/duoergun0729/adversarial_examples/blob/master/code/4-demo.ipynb
 #### 彩图，灰度图，二值化
 opencv显示RGB彩图。opencv默认格式为BGR，所以需要转换一下
 ```py
 import cv2
 from matplotlib import pyplot as plt
 img_path = input("Image_path")
 img = cv2.imread(img_path)
 show_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 plt.imshow(show_img)
 plt.show()
 print(img.shape)
 ```
 opencv转灰度图
 ```py
 show_grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 plt.imshow(show_grayimg, cmap = plt.cm.gray)
 plt.show()
 print(img.shape)
 ```
 opencv转二值图
 ```py
 ret, thresh = cv2.threshold(show_img, 220, 255, cv2.THRESH_BINARY)
 plt.imshow(thresh, cmap = plt.cm.gray)
 plt.show()
 ```
 关于cv2.threshold，原型为
 ```py
 cv2.threshold(src,x,y,Methods)
 ```
 ·src 指原始图像，该图像为灰度图
 
 ·x 指用来对像素值进行分类的阈值
 
 ·y 指当像素值高于阈值时应该被赋予的新的像素值
 
 ·Methods 指不同的阈值方法，包括：cv2.THRESH_BINARY cv2.THRESH_BINARY_INV cv2.THRESH_TRUNC cv2.THRESH_TOZERO cv2.THRESH_TOZERO_INV
 #### BMP,JPEG,GIF,PNG
 ·BMP 无损压缩

 ·JPEG 有损压缩

 ·GIF 基于LZW算法的连续色调的无损压缩，压缩率一般在50%左右。可制作动图

 ·PNG 能够提供长度比GIF小30%的无损压缩图像文件。
 ### 图像转换
 #### 仿射变换
 $$\left|\begin{matrix} x' \\y' \\ 1 \end{matrix}\right| = \left|\begin{matrix}a_{1}&a_{2}&t_{1} \\ a_{3}&a_{4}&t_{2} \\ 0 & 0 & 1 \end{matrix}\right| X \left|\begin{matrix}x \\
 y \\  1 \end{matrix}\right| $$
 其中$\left|\begin{matrix}a_{1}&a_{2} \\ a_{3}&a_{4} \end{matrix}\right|$代表旋转和缩放变化，$\left|\begin{matrix} t_{1} \\ t_{2} \end{matrix}\right|$代表平移变化$\left|\begin{matrix} a_{1}&a_{2}&t_{1} \\a_{3}&a_{4}&t_{2}\end{matrix}\right|$因此被称为仿射变化
 
 Opencv支持仿射变化
 ##### warpAffine函数
 ```py
 warpAffine(src, M, dsize, flags, borderMode, borderValue)
 ```
 ·src 输入变化前的图像
 
 ·M 仿射变化矩阵
 
 ·dsize 设置输出图像大小
 
 ·flags 设置插值方式，默认为线性插值
 
 ·borderMode 边界像素模糊
 
 ·borderValue 边界填充值，默认情况下为0，即填充黑色
 
 ##### getRotationMatrix2D函数
 ```py
 getRotationMatrix2D(center, angle, scale)
 ```
 ·center 旋转中心
 
 ·angle 旋转角度
 
 ·scale 图像缩放倍数
 ### 图像去噪
 高斯噪声（Gaussian noise）是指其概率密度分布服从高斯分布的一种噪声。盐噪声（Salt noise）为白色，椒噪声（Pepper noise）为黑色。
 
 叠加噪声可以通过skimage库，其提供了skimage.util.random_noise函数用于增加噪声
 ```py
 skimage.util.random_noise(image, mode = 'gaussion',seed = None,clip = True,mean,var)
 ```
 #### 中值滤波
 ```py
 cv2.medianBlur()
 ```
 #### 高斯滤波
 ```py
 cv2.GaussianBlur()
 ```
 #### 高斯双边滤波
 ```py
 bilateralFilter(src,n,sigmaColor,sigmaSpace,borderType)
 ```
 ·src：输入变化前图像
 ·n：过滤中每个像素邻域的直径范围
 ·sigmaColor：颜色空间过滤器的标准差值
 ·sigmaSpace：坐标空间中滤波器的标准差值
 ·borderType：用于推断图像外部像素的某种边界模式，默认值为 BORDER_DEFAULT
 
 ## Day5
 白盒攻击算法
 ### 对抗样本的基本原理
 从数学角度来描述对抗样本，输入数据x，分类器为f，对应的分类结果表示为f（x），假设存在一个非常小的扰动$\epsilon$，使得f(x+$\epsilon$) != f(x)，则$x+\epsilon$是一个对抗样本
 
 源代码：https://github.com/duoergun0729/adversarial_examples/blob/master/code/5-case1.ipynb
 
 通过datasets库生成样本数据。其中n_features为特征数，n_samples为生成的样本数量。
 ```py
 from sklearn import datasets
 from sklearn.preprocessing import MinMaxScaler
 import keras
 from keras.models import Sequential
 from keras.layers import Dense
 from keras.utils import to_categorical
 from keras.optimizers import RMSprop
 import numpy as np
 import matplotlib.pyplot as plt
 ```
 为了方便处理，把 样本归一化到(0，1)的数据，标签转换为独热编码
 ```py
 n_features = 1000
 x ,y = datasets.make_classification(n_samples = 4000, n_features = n_features, n_classes = 2, random_state = 0)
 # 转换成独热编码
 y = to_categorical()
 # 归一化到（0，1）数据
 x = MinMaxScaler().fit_transform(x)
 ```
 分类模型为多层感知机，输入层为1000，输出层为2，激活函数为softmax
 ```py
 model = Sequential()
 model.add(Dense(2,activation = 'softmax', input_shape = (n_features,)))
 model.compile(loss = 'categorical_crossentropy',optimizer = RMSprop(),metrics = ['accuracy'])
 ```
 打印模型结构
 ```py
 model.summary()
 ```
 对模型进行训练，批大小为16，训练30轮，准确稳定度在86%左右
 ```py
 model.fit(x,y,epochs = 30,batch_size = 16)
 ```
 针对第0号数据进行预测
 ```py
 x0 = x[0]
 y0 = y[0]
 x0 = np.expend_dims(x0, axis = 0)
 y0_predict = model.predict(x0)
 print(y0_predict)
 index = [0,1]
 labels = ["label 0", "label 1"]
 probability = y0_predict[0]
 ```
 增加一个0.01或者-0.01的扰动量进行预测
 ```py
 e = 0.01
 cost, gradients = grab_cost_and_gradients_from_model([x0, 0])
 n = np.sign(gradients)
 x0 += n * e
 y0_predict = model.predict(x0)
 print(y0_predict)
 ```
 ### 基于优化的对抗样本生成算法
 比对原始数据与对抗样本的差距，定义一个展现函数show_images_diff，主要参数有：
 · original_img: 原始数据
 · original_label: 原始数据的标签
 · adversarial_img: 对抗样本
 · adversarial_label: 对抗样本的标签
 ```py 
 def show_images_diff(original_img, original_label, adversarial_img, adversarial_label):
   if original_img.any() > 1.0:
    original_img = original_img/255.0
   if adversarial_img.any() > 1.0:
    adversarial_img = adversarial_img/255.0
   difference = adversarial_img - original_img
   # (-1,1) > (0,1)
   difference = difference / abs(difference).max()/2.0 + 0.5
 ```
 #### 使用PyTorch生成对抗样本（待完善）
 #### 使用Tensorflow生成对抗样本
 源代码：http://github.com/duoergun0729/adversarial_examples/blob/master/code/5-case2-tensorflow-pb.ipynb
 
