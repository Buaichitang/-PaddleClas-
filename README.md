# 基于PaddleClas的食物分类（大标题）

运用PaddleClas对进行食物分类

# 一、项目背景
将食物进行分类，粗分类后的食物便于人们对其热量进行下一步的判断

# 二、数据集简介

此项目使用了[AI训练营]提供的食物五分类数据集，共有5000张图片格式为.jpg

## 1.数据加载和预处理

```python
# 项目挂载的数据集先解压出来，待解压完毕，刷新后可发现左侧文件夹根目录出现五个zip
!unzip -oq /home/aistudio/data/data103736/五种图像分类数据集.zip
# 解压完毕左侧出现文件夹，即为需要分类的文件
!unzip -oq /home/aistudio/食物5分类.zip

```

训练集样本量: 60000，验证集样本量: 10000


## 2.数据集查看


```python
# 查看结构，正为一个类别下有一系列对应的图片
!tree foods/

```


# 三、模型选择和开发

    第一层：卷积层1，输入为 224 × 224 × 3 224 \times 224 \times 3 224×224×3的图像，卷积核的数量为96，论文中两片GPU分别计算48个核; 卷积核的大小为 11 × 11 × 3 11 \times 11 \times 3 11×11×3; stride = 4, stride表示的是步长， pad = 0, 表示不扩充边缘;
    卷积后的图形大小是怎样的呢？
    wide = (224 + 2 * padding - kernel_size) / stride + 1 = 54
    height = (224 + 2 * padding - kernel_size) / stride + 1 = 54
    dimention = 96
    然后进行 (Local Response Normalized), 后面跟着池化pool_size = (3, 3), stride = 2, pad = 0 最终获得第一层卷积的feature map
    最终第一层卷积的输出为
    第二层：卷积层2, 输入为上一层卷积的feature map， 卷积的个数为256个，论文中的两个GPU分别有128个卷积核。卷积核的大小为： 5 × 5 × 48 5 \times 5 \times 48 5×5×48; pad = 2, stride = 1; 然后做 LRN， 最后 max_pooling, pool_size = (3, 3), stride = 2;
    第三层：卷积3, 输入为第二层的输出，卷积核个数为384, kernel_size = ( 3 × 3 × 256 3 \times 3 \times 256 3×3×256)， padding = 1, 第三层没有做LRN和Pool
    第四层：卷积4, 输入为第三层的输出，卷积核个数为384, kernel_size = ( 3 × 3 3 \times 3 3×3), padding = 1, 和第三层一样，没有LRN和Pool
    第五层：卷积5, 输入为第四层的输出，卷积核个数为256, kernel_size = ( 3 × 3 3 \times 3 3×3), padding = 1。然后直接进行max_pooling, pool_size = (3, 3), stride = 2;
    第6,7,8层是全连接层，每一层的神经元的个数为4096，最终输出softmax为1000,因为上面介绍过，ImageNet这个比赛的分类个数为1000。全连接层中使用了RELU和Dropout。



## 1.模型组网

![](https://ai-studio-static-online.cdn.bcebos.com/51bb295ad9b74f10af17a088eba39414290784b639854919976c868b61d17958)


## 2.模型训练


```python
#AlexNet对食物进行分类
!python3 tools/train.py \
    --config ./ppcls/configs/ImageNet/AlexNet/AlexNet.yaml
```

    2021/08/15 15:59:37] root INFO: [Train][Epoch 1/20][Iter: 0/36]lr: 0.10000, CELoss: 1.79073, loss: 1.79073, top1: 0.17969, top5: 0.82812, batch_cost: 1.12537s, reader_cost: 1.09916, ips: 113.74039 images/sec, eta: 0:13:30
[2021/08/15 15:59:38] root INFO: [Train][Epoch 1/20][Iter: 1/36]lr: 0.10000, CELoss: 1.78895, loss: 1.78895, top1: 0.19922, top5: 0.91406, batch_cost: 0.96297s, reader_cost: 0.93762, ips: 132.92237 images/sec, eta: 0:11:32
[2021/08/15 15:59:39] root INFO: [Train][Epoch 1/20][Iter: 2/36]lr: 0.10000, CELoss: 1.78533, loss: 1.78533, top1: 0.21875, top5: 0.94271, batch_cost: 0.94977s, reader_cost: 0.92510, ips: 134.76967 images/sec, eta: 0:11:21
[2021/08/15 15:59:39] root INFO: [Train][Epoch 1/20][Iter: 3/36]lr: 0.10000, CELoss: 1.78184, loss: 1.78184, top1: 0.22461, top5: 0.95703, batch_cost: 0.93078s, reader_cost: 0.90637, ips: 137.51933 images/sec, eta: 0:11:07
[2021/08/15 15:59:40] root INFO: [Train][Epoch 1/20][Iter: 4/36]lr: 0.10000, CELoss: 1.77832, loss: 1.77832, top1: 0.20625, top5: 0.96562, batch_cost: 0.91827s, reader_cost: 0.89412, ips: 139.39288 images/sec, eta: 0:10:57


# 四、总结与升华

要对所获得的数据集进行处理，划分为训练集、测试集，训练的过程中很主要的步骤为调整参数，通过设置不同的迭代次数，会使得loss值发生变化，但是并不是迭代次数越多loss值越小。


# 个人简介

我在AI Studio上获得白银等级，点亮2个徽章，来互关呀~ https://aistudio.baidu.com/aistudio/personalcenter/thirdview/512456