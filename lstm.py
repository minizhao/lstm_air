import os
import sys
import csv
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import  Variable
from tqdm import tqdm

class DataSet(object):
    """数据处理的类"""
    def __init__(self, data_dir="beijing"):
        super(DataSet, self).__init__()
        self.data_dir = data_dir
        self.org_data=[]
        self.load_data()
        self.gen_train_data()

    def load_data(self):
        for file in sorted(os.listdir(self.data_dir)):
            if "csv"  not in file:
                continue
            with open(os.path.join(self.data_dir,file)) as f:
                lines=csv.reader(f)
                header=next(lines)

                #type之后的是数据值
                type_idx=header.index('type')

                temp_list=[]
                for line in lines:
                    data_values=line[type_idx+1:]
                    if len(data_values)!=35:
                        continue
                    temp_list.append(data_values)
                    #得到的原始数据
                    if len(temp_list)==5:
                        self.org_data.append(temp_list)
                        temp_list=[]

    def gen_train_data(self):
        #先把数据按照不同的type分离出来
        self.splited_data=[]
        for item in self.org_data:
            pm2_5,pm2_5_24h,pm_10,pm_10_24h,aqi=item
            for idx in range(len(pm2_5)):
                self.splited_data.append((pm2_5[idx],pm2_5_24h[idx],\
                    pm_10[idx],pm_10_24h[idx],aqi[idx]))

        # 滑动窗口把数据切分成inp,target
        self.data=[]
        seg_len=25
        for idx in range(len(self.splited_data)):
            seg=self.splited_data[idx:idx+seg_len]
            if len(seg)!=seg_len:
                continue
            inp,target=seg[:-5],[x[0] for x in seg[-5:]]
            self.data.append((inp,target))

        np.random.shuffle(self.data)
        self.train_data=self.data[:int(len(self.data)*0.8)]
        self.test_data=self.data[int(len(self.data)*0.8):]


class Model(nn.Module):
    """模型类"""
    def __init__(self, input_size=5,hidden_size=128,num_layers=1,batch_size=32):
        super(Model, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.lstm= nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)

        # 用过去20小时个预测未来5个小时的数据数据，输出是5维的
        self.output=nn.Linear(20*hidden_size,5)

    def forward(self,input):
        hidden = (Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size)),
                  Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size))
                 )
        out_enc,_= self.lstm(input, hidden)
        out_enc=self.output(out_enc.contiguous().view(self.batch_size,-1))
        return out_enc

class Train(object):
    """训练模型类"""
    def __init__(self,model,optimizer,criterion,train_data,test_data,batch_size=32):
        super(Train, self).__init__()
        self.model = model
        self.train_data=train_data
        self.test_data=test_data
        self.batch_size=batch_size
        self.optimizer=optimizer
        self.criterion=criterion

    def train(self,epochs=10):
        for idx in tqdm(range(len(self.train_data))):
            batch_data=self.train_data[idx:idx+self.batch_size]
            inp,target=[x[0] for x in batch_data],[x[1] for x in batch_data]
            inp,target=self.conver_tensor(inp,target)
            output=lstm(inp)
            loss=criterion(output,target)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def conver_tensor(self,inp,target):
        inp=np.array(inp)
        target=np.array(target)
        inp[inp=='']=0
        target[target=='']=0
        inp=np.array(inp).astype(np.float)
        target=np.array(target).astype(np.float)
        inp=torch.from_numpy(inp).type(torch.FloatTensor)
        target=torch.from_numpy(target).type(torch.FloatTensor)
        return inp,target


if __name__ == '__main__':

    dataSet=DataSet()
    lstm=Model()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
    train=Train(lstm,optimizer,criterion,dataSet.train_data,dataSet.test_data)
    train.train(10)
