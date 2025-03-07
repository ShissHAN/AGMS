import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
used_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader
import time
import argparse
import model
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import scipy.io as sio
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='hyperspectral target detection')
parser.add_argument('--data_dir', type=str, default='./data/')
parser.add_argument('--data_name', type=str, default='AVIRIS')# AVIRIS / Nuance-CRI
parser.add_argument("--num_epoch",type=int,default=50,help="epoch number of training")
parser.add_argument("--batch_size",type=int,default=32,help="batch size of training")
parser.add_argument("--lr",type=float,default=1e-3,help="adam: learning rate")
parser.add_argument('--patience',type=int, default=7, help="patience of early stop")
parser.add_argument("--margin",type=float,default=0.6,help="the margin value of ContrastiveLoss")
args = parser.parse_args()
print(args)

class Data_set(Dataset):
    def __init__(self,data,label):
        super().__init__()
        self.len = data.shape[0]
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index],self.label[index]

    def __len__(self):
        return self.len

def data_reader(data_path,device):
    mat_contents = sio.loadmat(data_path)
    prior_target = mat_contents['S'].astype(float)
    prior_target = np.transpose(prior_target, (1, 0))
    data_3d = mat_contents['X'].astype(float)
    mask = mat_contents['mask'].astype(int)

    prior_target = torch.cuda.FloatTensor(prior_target).to(device)
    data_3d = torch.cuda.FloatTensor(data_3d).to(device)
    mask = torch.IntTensor(mask).to(device)

    [h,w,bands] = data_3d.shape
    data_2d = torch.reshape(data_3d,[h*w,bands])
    mask = torch.transpose(mask,0,1)

    prior_target = torch.mean(prior_target,dim=0,keepdim=True)
    print('prior_target.shape:',prior_target.shape,data_2d.shape,mask.shape)
    data = torch.cat((prior_target,data_2d), dim=0)
    data_min = torch.min(data)
    data_max = torch.max(data)
    prior_target = (prior_target-data_min) / (data_max-data_min)
    data_2d = (data_2d-data_min) / (data_max-data_min)
    return prior_target, data_2d, mask


def data_loader(prior_target, gen_target, selected_back):
    sim_label = torch.unsqueeze(torch.ones(gen_target.shape[0]).to(prior_target.device),dim=1)
    print(gen_target.shape,selected_back.shape)
    disim_label = torch.unsqueeze(torch.zeros(selected_back.shape[0]).to(prior_target.device),dim=1)
    trainLabel = torch.cat((sim_label,disim_label),dim=0)
    gen_target = gen_target.to(prior_target.device)
    selected_back = selected_back.to(prior_target.device)
    trainData = torch.cat((gen_target,selected_back),dim=0)
    train_data = Data_set(trainData,trainLabel)
    print(gen_target.shape, selected_back.shape)
    return train_data


def init_weights(m):
    if isinstance(m,nn.Conv1d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m,nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m,nn.Linear):
        nn.init.normal_(m.weight.data,mean=0,std=0.01)
        m.bias.data.zero_()

def get_train_samples(prior_target,image,args):
    #读取文件夹里的背景样本
    filename = "./resultAV_bg/decoder/x_decoder100"
    selected_back1 = loadmat(file_name=filename)
    selected_back = selected_back1['x_decoder'].astype(float)
    selected_back = torch.from_numpy(selected_back)

    #读取文件夹里的目标生成样本
    filename =  "gen_target_AV"
    gen_target1 = loadmat(file_name=filename)
    gen_target = gen_target1['gen_target_AV'].astype(float)
    '''min_value = np.min(gen_target)
    max_value = np.max(gen_target)
    gen_target = (gen_target - min_value) / (max_value - min_value)'''
    gen_target = torch.from_numpy(gen_target)
    gen_target = torch.Tensor(gen_target)
    return gen_target, selected_back

def _init_():
    if not os.path.exists('pth'):
        os.makedirs('pth')
    if not os.path.exists('result'): 
        os.makedirs('result')

def standard(X):
    max_value = np.max(X)
    min_value = np.min(X)
    if max_value == min_value:
        return X
    return (X - min_value) / (max_value - min_value)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(model,prior_target,train_loader,criterion,optimizer,num_epoch,patience):
    writer = SummaryWriter(log_dir = './runs/loss')
    best_loss = float('inf')
    early_stop_counter = 0
    train_loss = 0
    model.train()
    for epoch in range(num_epoch):
        for data,label in train_loader:
            optimizer.zero_grad()#将梯度置为0
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dtype = torch.float32  # 指定数据类型为 float32
            data = data.to(device, dtype=dtype)
            #data = torch.cuda.FloatTensor(data).to(device)
            out1,out2 = model(prior_target,data)
            loss = criterion(out1,out2,label,model)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        if (epoch+1)%1 == 0:
            print('Epoch[{}/{}],      train_loss:  {:.6f}'.format(epoch+1, num_epoch, train_loss))
            writer.add_scalars('train_',{'loss':loss},epoch+1)
        if train_loss < best_loss:
            best_loss = train_loss
            early_stop_counter = 0
            torch.save(model.state_dict(),'./pth/model.pt')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early Stopping!")
                break
    writer.close()


def detec(model,prior_target,detec_data,height, width):
    model.eval()
    model.load_state_dict(torch.load('./pth/model.pt'))
    detec_result = []
    with torch.no_grad():
        output1,output2 = model(prior_target,detec_data)
        sim = torch.mean(F.cosine_similarity(output1,output2,dim=2),dim=1,keepdim=True)
        detec_result.append(sim)
    detec_result = torch.squeeze(torch.stack(detec_result,dim=0))
    detec_result = torch.reshape(detec_result,(width,height))
    #detec_result = torch.transpose(detec_result,0,1)#detec_result.shape=[height,width]
    return detec_result


if __name__=='__main__':
    _init_()
    setup_seed(3333)
    # Defining the model, loss, optimizer
    net = model.SiameseNet(model.EmbeddingNet()).to(used_device)
    net.apply(init_weights)
    criterion = model.ContrastiveLoss(margin=args.margin).to(used_device)
    optimizer = optim.Adam(net.parameters(),lr=args.lr)

    # Loading data
    prior_target, image, mask = data_reader(data_path=os.path.join(args.data_dir, '{}.mat'.format(args.data_name)),device=used_device)
    height, width = mask.shape
    gen_target, selected_back = get_train_samples(prior_target,image,args)
    print(type(gen_target))
    print(type(selected_back))
    print(gen_target.shape)
    #输出gen_target的类型
    #gen_target = torch.FloatTensor(gen_target).to(device)
    train_set = data_loader(prior_target,gen_target,selected_back)
    trainData = DataLoader(train_set,batch_size=args.batch_size,shuffle=True)

    # Training and detection
    start = time.time()
    train(net,prior_target,trainData,criterion,optimizer,args.num_epoch,args.patience)
    detec_result = detec(net,prior_target,image,height,width)
    print(detec_result.shape)

    #可视化
    tensor_cpu = detec_result.cpu()
    detec_result = tensor_cpu.detach().numpy()
    #savemat('./result/AVIRIS_OUR_probability.mat', {'probability': detec_result})
    plt.imshow(detec_result, cmap='gray'), plt.title('detec_result')
    plt.show()

    # 将detec_result变成一维数组
    ## calculate the AUC value
    detec_result = standard(detec_result)
    detec_result = np.clip(detec_result, 0, 1)
    detec_result = np.reshape(detec_result, [-1, 1], order='F')
    tensor_cpu = mask.cpu()
    mask = tensor_cpu.detach().numpy()
    mask = np.reshape(mask, [-1, 1], order='F')
    y_p = detec_result.T
    y_p = np.reshape(y_p, (-1))
    auc = metrics.roc_auc_score(mask, y_p)
    print('AUC=',auc)
    print(time.time()-start)

