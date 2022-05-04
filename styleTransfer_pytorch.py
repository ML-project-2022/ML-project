import numpy as np
from torchvision.models import vgg19
from torch import nn
from torchvision.utils import save_image
import torch
import cv2
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 读取图片
def load_image(path):
    image = cv2.imread(path)  # 打开图片
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换通道，因为opencv默认读取格式为BGR，转换为RGB格式
    image = torch.from_numpy(image).float() / 255  # 数值归一化操作
    image = image.permute(2, 0, 1).unsqueeze(0)  # 换轴，（H,W,C）转换为（C,H,W），并做升维处理。
    return image

# 定义损失函数
def get_gram_matrix(features_map): #计算gram矩阵
    n, c, h, w = features_map.shape
    if n == 1:
        features_map = features_map.reshape(c, h * w)
        gram_matrix = features_map@features_map.T
        return gram_matrix
    else:
        raise ValueError('Can not process more than one picture')

def style_loss(feature_bank_x,feature_bank_style):
    E=0
    n_layer=len(feature_bank_style)
    w=1/n_layer
    for i, feature in enumerate(feature_bank_style):
        shape=feature_bank_x[i].shape
        C = int(shape [1])
        H = int(shape[2])
        W = int(shape[3])
        G_x=get_gram_matrix(feature_bank_x[i])
        G_s = get_gram_matrix(feature)
        loss_func=nn.MSELoss().cuda()
        E += w * loss_func(G_x,G_s)/ (4 * C**2 * H**2 * W**2)
    return E

def content_loss(out_x, out_content):
    loss_func=nn.MSELoss().cuda()
    return loss_func(out_x, out_content)

#建立模型
class VGG19(nn.Module): #vgg_19 model
    def __init__(self):
        super(VGG19, self).__init__()
        self.indexes=[0,]
        vgg_model=vgg19()
        pre_trained=torch.load("vgg_para/vgg19.pth")
        vgg_model.load_state_dict(pre_trained)
        self.features=vgg_model.features
        for i,layer in enumerate(self.features):
            if isinstance(layer,nn.ReLU):
                self.indexes.append(i)
        selected_layer=[0,1,3,5,9,10,13] # 选择用来计算损失函数的ReLU层
        self.indexes=np.array(self.indexes)[selected_layer]

    def forward(self,input):
        features_bank=[]
        out=input
        n=len(self.indexes)
        for i in range(1,n):
            out=self.features[self.indexes[i-1]:self.indexes[i]](out)
            features_bank.append(out)
        out=features_bank[-2]
        return features_bank, out


class GNet(nn.Module): # 要训练的model
    def __init__(self, image):
        super(GNet, self).__init__()
        self.image_g = nn.Parameter(image.detach().clone()) # 从一张图片开始

    def forward(self):
        return self.image_g.clamp(0, 1)

#训练模型
content_path='images/content0.jpg'
style_path='style/style1.jpg'
content_img=load_image(content_path).cuda()
style_img=load_image(style_path).cuda()
g_net=GNet(content_img).cuda()
vgg_net= VGG19().cuda()
with torch.no_grad():
    features_bank_style, out_style=vgg_net(style_img)
    features_bank_content, out_content=vgg_net(content_img)

def train_loop(alpha,beta ,learning_rate):
    image_x = g_net()
    features_bank_x, out_x= vgg_net(image_x)
    loss_s=style_loss(features_bank_x,features_bank_style)
    loss_c=content_loss(out_x, out_content)
    loss_total=alpha*loss_s+beta*loss_c
    optimizer = torch.optim.Adam(g_net.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
    return image_x

epoches=5001
for t in tqdm(range(epoches)):
    image_x=train_loop(1e-3, 1, 0.01)
    if t % 1000 == 0:
        save_image(image_x, f'{t/100}.jpg', padding=0, normalize=True, value_range=(0, 1))
