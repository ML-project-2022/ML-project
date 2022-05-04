# 卷积神经网络与风格迁移

尽管区分图像的语义和风格在传统图像处理方法中是困难的, 但我们可以比较容易地用卷积神经网络来分离图片的纹理和内容. 考虑卷积核的大小, 当卷积核比较大的时候, 我们提取的是整个图像粗粒度的信息(比如风格), 而当卷积核比较小的时候, 则会得到图片中像素级别的信息. 
因此风格在直观上来说比较像是用大卷积核提取特征后的结果, 而内容在直观上来说是小卷积核提取特征后的结果. 

在具体的优化中, 我们用特征图的欧式距离来表征内容损失(这是常见的处理方法). 用特征图gram矩阵的欧式距离来表征风格损失(度量的是主要特征间的差异). 在内容损失和风格损失相加时很明显有一个权重超参数. 这将决定风格化的明显程度

 

事实上
---------------------

Firstly, you need these package of python: tensorflow, numpy, scipy, pillow

just use these commands: pip install tensorflow, pip install numpy, pip install scipy, pip install pillow

Secondly, you need the pre-trained model of vgg-19, you can download the model from this [address](https://pan.baidu.com/s/1CO-A2GOoym7eCw0hQsvEyw), whose model had removed the fully connected layer to reduce parameters.After you download the model, you should put it in this folder named "vgg_para".

Method
-------

![algorithm](https://github.com/MingtaoGuo/Style-transfer-with-neural-algorithm/raw/master/method/method.jpg)

We can see the image above that is from the paper, it shows a simple way to synthesize an image from other style.In this method, x is the variable which we want to update, and it is also an synthesized image as the final result. The squared error is used to control the content which makes the synthesized image is similar to the original content image, and the Gram matrix is used to control the style which makes the synthesized image has the similar style with original style image.

Result
-----------

![content0](https://github.com/MingtaoGuo/Style-transfer-with-neural-algorithm/raw/master/images/result.jpg)

This result's parameter: alpha 1e-5, beta 1.0, width 512, height 512, optimizer: L-BFGS, iteration of L-BFGS 500, the result of Adam is not very well.
