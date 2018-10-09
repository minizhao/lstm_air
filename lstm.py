import os
import sys
import csv
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import  Variable
from tqdm import tqdm
from visdom import Visdom
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree
from sklearn.metrics import mean_squared_error
np.random.seed(100)
CUDA=torch.cuda.is_available()

class DataSet(object):
    """数据处理的类"""
    def __init__(self, data_dir="beijing"):
        super(DataSet, self).__init__()
        self.data_dir = data_dir
        self.org_data=[]
        self.postions=[
        '东四', '天坛', '官园', '万寿西宫', '奥体中心', '农展馆', '万柳', '北部新区',
        '植物园', '丰台花园', '云岗', '古城', '房山', '大兴', '亦庄', '通州', '顺义',
        '昌平', '门头沟', '平谷', '怀柔', '密云', '延庆', '定陵', '八达岭', '密云水库',
        '东高村', '永乐店', '榆垡', '琉璃河', '前门', '永定门内', '西直门北', '南三环', '东四环'
        ]
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
        #把数据按照不同的地方串起来
        self.postions_data={}
        for item in self.org_data:
            pm2_5,pm2_5_24h,pm_10,pm_10_24h,aqi=item
            for idx in range(len(pm2_5)):
                pos=self.postions[idx]
                self.postions_data[pos]=self.postions_data.get(pos,[])
                self.postions_data[pos].append((pm2_5[idx],pm2_5_24h[idx],\
                    pm_10[idx],pm_10_24h[idx],aqi[idx]))


        # 滑动窗口把数据切分成inp,target
        self.data=[]
        self.vaild_data={'天坛':[],'通州':[],'前门':[]}

        for pos,pos_data in self.postions_data.items():

            seg_len=25
            for idx in range(len(pos_data)):
                seg=pos_data[idx:idx+seg_len]

                # 去除一些异常数据
                if len(seg)!=seg_len:
                    continue
                inp,target=seg[:-5],[x[0] for x in seg[-5:]]

                ishas_nan=False
                for item in inp:
                    if '' in item:
                        ishas_nan=True
                        break
                if ishas_nan:
                    continue

                if len([x for x in target if x==''])>0:
                    continue
                if len([x for x in target if float(x)>1000])>0:
                    continue

                # 保留三个测试地方的数据
                if pos not in self.vaild_data.keys():
                    self.data.append((inp,target))
                else:
                    self.vaild_data[pos].append((inp,target))

        np.random.shuffle(self.data)
        self.train_data=self.data[:int(len(self.data)*0.3)]
        self.test_data=self.data[int(len(self.data)*0.98):]

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
        self.batch_size=input.size(0)
        if CUDA:
            hidden = (Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size)).cuda(),
                      Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size)).cuda()
                     )
        else:
            hidden = (Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size)),
                      Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size))
                     )
        out_enc,_= self.lstm(input, hidden)
        out_enc=self.output(out_enc.contiguous().view(self.batch_size,-1))
        return out_enc



class compared_models(object):
    """对比方法"""
    def __init__(self,dataSet):
        super(compared_models, self).__init__()
        self.LinearR=LinearRegression()
        self.DeciR= tree.DecisionTreeRegressor()
        self.GbdtR= GradientBoostingRegressor(n_estimators=70)
        self.dataSet=dataSet
        self.trs_data()
        self.train()


    def trs_data(self):
        """
        数据变换函数
        """
        self.train_x=[]
        self.train_y=[]
        for item in self.dataSet.train_data:
            self.train_x.append(item[0])
            self.train_y.append(item[1])
        nums_samples=len(self.train_x)
        self.train_x=np.array(self.train_x).astype(float).reshape(nums_samples,-1)
        self.train_y=np.array(self.train_y).astype(float)[:,0]

        assert self.train_x.shape[0]==self.train_y.shape[0]

    def cover_data(self,inp_data):
        # 将预测数据进行格式变化
        self.pred_inp=[]
        for inp,target in inp_data:
            self.pred_inp.append(inp)
        nums_samples=len(self.pred_inp)
        self.pred_inp=np.array(self.pred_inp).astype(float).reshape(nums_samples,-1)



    def train(self):
        self.LinearR.fit(self.train_x, self.train_y)
        self.DeciR.fit(self.train_x, self.train_y)
        self.GbdtR.fit(self.train_x, self.train_y)


    def pred(self,inp_data):
        self.cover_data(inp_data)
        LinearRst=self.LinearR.predict(self.pred_inp)
        DeciRRst=self.DeciR.predict(self.pred_inp)
        GbdtRst=self.GbdtR.predict(self.pred_inp)
        return LinearRst,DeciRRst,GbdtRst

class Train(object):
    """训练模型类"""
    def __init__(self,model,optimizer,criterion,train_data,test_data,cm,batch_size=32):
        super(Train, self).__init__()
        self.model = model
        self.train_data=train_data
        self.test_data=test_data
        self.batch_size=batch_size
        self.optimizer=optimizer
        self.criterion=criterion
        self.cm=cm
        self.every_save=10000

    def train(self,epochs=100):
        viz = Visdom()
        line = viz.line(np.arange(2))

        step_p=[]
        train_loss_p=[]
        test_loss_p=[]

        best=np.inf
        for epoch in range(epochs):
            train_loss=[]
            test_loss=[]
            for idx in tqdm(range(0,len(self.train_data),self.batch_size)):
                batch_data=self.train_data[idx:idx+self.batch_size]
                inp,target=[x[0] for x in batch_data],[x[1] for x in batch_data]
                inp,target=self.conver_tensor(inp,target)
                if inp.size(0)!=self.batch_size:
                    continue
                output=self.model(inp)
                loss=criterion(output,target)
                train_loss.append(loss.data[0])
                # print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mean_test_loss=self.test()
            if mean_test_loss<best:
                torch.save(self.model,'model.pt')
                best=mean_test_loss


            print("epoch :{},train mean loss:{},test mean loss:{}".format(epoch,np.mean(train_loss),mean_test_loss))

            train_loss_p.append(np.mean(train_loss))
            test_loss_p.append(mean_test_loss)
            step_p.append(epoch)

            viz.line(
                 X=np.column_stack((np.array(step_p), np.array(step_p))),
                 Y=np.column_stack((np.array(train_loss_p),np.array(test_loss_p))),
                 win=line,
                opts=dict(legend=["Train_mean_loss", "Test_mean_loss"]))

    def conver_tensor(self,inp,target):
        inp=np.array(inp)
        target=np.array(target)
        inp[inp=='']=0
        target[target=='']=0
        inp=np.array(inp).astype(np.float)
        target=np.array(target).astype(np.float)
        if CUDA:
            inp=torch.from_numpy(inp).type(torch.FloatTensor).cuda()
            target=torch.from_numpy(target).type(torch.FloatTensor).cuda()
        else:
            inp=torch.from_numpy(inp).type(torch.FloatTensor)
            target=torch.from_numpy(target).type(torch.FloatTensor)
        return inp,target

    def test(self):
        loss_list=[]
        with torch.no_grad():
            for idx in tqdm(range(0,len(self.test_data),self.batch_size)):
                batch_data=self.train_data[idx:idx+self.batch_size]
                inp,target=[x[0] for x in batch_data],[x[1] for x in batch_data]
                inp,target=self.conver_tensor(inp,target)
                if inp.size(0)!=self.batch_size:
                    continue
                output=self.model(inp)
                loss=criterion(output,target)
                loss_list.append(loss.data[0])
        return np.mean(loss_list)

    def pred(self,inp_data):
        preds=[]
        with torch.no_grad():
            for inp,target in inp_data:
                inp,target=self.conver_tensor(inp,target)
                inp=inp.unsqueeze(0)
                output=self.model(inp)
                preds.append(output.data[0].numpy()[0])
        return preds

    def relat_acc(self,y_true, y_pred):
        relat_delta=np.abs(y_true-y_pred)/y_true
        print(len([x for x in relat_delta if x<0.1])/len(relat_delta))
        print(len([x for x in relat_delta if x<0.2])/len(relat_delta))
        print(len([x for x in relat_delta if x<0.3])/len(relat_delta))

    def polt(self,dataset):
        """
        画出测试图函数
        """
        start=1323
        """ '天坛','通州','前门' """
        plot_data=dataset.vaild_data["通州"][start:start+480]
        preds_lstm=self.pred(plot_data)

        preds_LinearR,preds_DeciR,preds_GbdtR=self.cm.pred(plot_data)
        x = np.array(list(range(len(plot_data))))
        lable = np.array([float(x[1][0]) for x in plot_data])

        print(mean_squared_error(lable, preds_lstm))
        self.relat_acc(lable, preds_lstm)
        print('-'*30)

        print(mean_squared_error(lable, preds_LinearR))
        self.relat_acc(lable, preds_LinearR)
        print('-'*30)

        print(mean_squared_error(lable, preds_DeciR))
        self.relat_acc(lable, preds_DeciR)
        print('-'*30)

        print(mean_squared_error(lable, preds_GbdtR))
        self.relat_acc(lable, preds_GbdtR)
        print('-'*30)

        sys.exit(0)



        plt.subplot(411)
        plt.plot(x,lable,label='Actual Value',color='red')
        plt.plot(x,preds_LinearR,label='SVM',color='blue')
        plt.ylabel('PM2.5')
        plt.xlabel('Timeline(hour)')
        plt.legend(loc='upper left')

        plt.subplot(412)
        plt.plot(x,lable,label='Actual Value',color='red')
        plt.plot(x,preds_DeciR,label='ARIMA',color='y')
        plt.ylabel('PM2.5')
        plt.xlabel('Timeline(hour)')
        plt.legend(loc='upper left')

        plt.subplot(413)
        plt.plot(x,lable,label='Actual Value',color='red')
        plt.plot(x,preds_GbdtR,label='GBDT',color='peru')
        plt.ylabel('PM2.5')
        plt.xlabel('Timeline(hour)')
        plt.legend(loc='upper left')

        plt.subplot(414)
        plt.plot(x,lable,label='Actual Value',color='red')
        plt.plot(x,preds_lstm,label='LSTM',color='green')
        plt.ylabel('PM2.5')
        plt.xlabel('Timeline(hour)')
        plt.legend(loc='upper left')


        plt.show()



if __name__ == '__main__':

    dataSet=DataSet()
    batch_size=128

    cm=compared_models(dataSet)

    print("len of train dataset is {}".format(len(dataSet.train_data)))
    print("len of test dataset is {}".format(len(dataSet.test_data)))
    if not os.path.isfile("model.pt"):
        if CUDA:
            lstm=Model(batch_size=batch_size).cuda()
        else:
            lstm=Model(batch_size=batch_size)
    else:
        lstm=torch.load("model.pt")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
    train=Train(lstm,optimizer,criterion,dataSet.train_data,dataSet.test_data,cm,batch_size)
    # train.train()
    train.polt(dataSet)
    # print(train.test())
