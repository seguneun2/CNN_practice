#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[20]:


def ConvBlock(in_dim, out_dim):
    model =nn.Sequential(
        nn.Conv2d(in_dim , out_dim, kernel_size = 3, stride=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.2, inplace = True))
    return model


# In[31]:


def ConvTransBlock(in_dim, out_dim):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size = 3, stride =2),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.2, inplace = True))
    return model


# In[22]:


def Maxpool():
    pool = nn.MaxPool2d(kernel_size = 2, stride=2, padding=0)
    return pool


# In[23]:


def cropping(bigger, smaller):
    diff_h = bigger.shape[2] - smaller.shape[2]
    diff_w = bigger.shape[3] - smaller.shape[3]
    cropped_bigger = nn.functional.pad(bigger, [0, -diff_w, 0, -diff_h])
    return cropped_bigger


# In[24]:


def ConvBlock2X(in_dim, out_dim):
    model = nn.Sequential(
        ConvBlock(in_dim, out_dim),
        ConvBlock(out_dim, out_dim))
    return model


# In[44]:


class Unet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filter):
        super(Unet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter  # 중간중간 사용할 conv filter수

        self.down_1 = ConvBlock2X(self.in_dim, self.num_filter)
        self.pool_1 = Maxpool()
        self.down_2 = ConvBlock2X(self.num_filter, self.num_filter*2)
        self.pool_2 = Maxpool()
        self.down_3 = ConvBlock2X(self.num_filter*2, self.num_filter*4)
        self.pool_3 = Maxpool()
        self.down_4 = ConvBlock2X(self.num_filter*4, self.num_filter*8)
        self.pool_4 = Maxpool()
        
        self.bridge = ConvBlock2X(self.num_filter*8, self.num_filter*16)
        
        self.trans_1 = ConvTransBlock(self.num_filter*16, self.num_filter*8) 
        # concat 해서 filter 2배.
        self.up_1 = ConvBlock2X(self.num_filter*16, self.num_filter*8)
        self.trans_2 = ConvTransBlock(self.num_filter*8, self.num_filter*4)
        self.up_2 = ConvBlock2X(self.num_filter*8, self.num_filter*4)
        self.trans_3 = ConvTransBlock(self.num_filter*4, self.num_filter*2)
        self.up_3 = ConvBlock2X(self.num_filter*4, self.num_filter*2)
        self.trans_4 = ConvTransBlock(self.num_filter*2, self.num_filter*1)
        self.up_4 = ConvBlock2X(self.num_filter*2, self.num_filter)
        
        self.out = nn.Sequential(
            nn.Conv2d(self.num_filter, self.out_dim, 1),
            nn.LeakyReLU(0.2, inplace = True))

    
    def forward(self, x):
        down_x1= self.down_1(x)
        pool_x1= self.pool_1(down_x1)
        down_x2= self.down_2(pool_x1)
        pool_x2= self.pool_2(down_x2)
        down_x3= self.down_3(pool_x2)
        pool_x3= self.pool_3(down_x3)
        down_x4= self.down_4(pool_x3)
        pool_x4= self.pool_4(down_x4)
        
        bridge = self.bridge(pool_x4)
        
        trans_x1= self.trans_1(bridge)
        cropping_x1 = cropping(down_x4, trans_x1) 
        concat_x1= torch.cat([trans_x1, cropping_x1], dim = 1) ##cropping? padiing?
        up_x1= self.up_1(concat_x1)
        trans_x2= self.trans_2(up_x1)
        cropping_x2 = cropping(down_x3, trans_x2) 
        concat_x2= torch.cat([trans_x2, cropping_x2], dim = 1)
        up_x2= self.up_2(concat_x2)
        trans_x3= self.trans_3(up_x2)
        cropping_x3 = cropping(down_x2, trans_x3)
        concat_x3= torch.cat([trans_x3, cropping_x3], dim = 1)
        up_x3= self.up_3(concat_x3)
        trans_x4= self.trans_4(up_x3)
        cropping_x4 = cropping(down_x1, trans_x4)
        concat_x4= torch.cat([trans_x4, cropping_x4], dim = 1)
        up_x4= self.up_4(concat_x4)
        
        out= self.out(up_x4)
        
        return out
        
        
        


# In[45]:


# in_dim=3
# out_dim=1
# num_filter=64


# # In[46]:


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Unet(in_dim,out_dim,num_filter).to(device)


# # In[47]:


# #print(model)


# # In[48]:


# import torchsummary


# # In[49]:


# torchsummary.summary(model,(3,572,572))  #input =  (batch, c, h, w)


# In[ ]:





# In[ ]:





# In[ ]:




