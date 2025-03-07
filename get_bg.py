"""
聚类得到初始背景
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import scipy.io as sio
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
used_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='hyperspectral target detection')
parser.add_argument('--data_dir', type=str, default='./data/')
parser.add_argument('--data_name', type=str, default='AVIRIS')# AVIRIS / Nuance-CRI / HYDICE-150-150-162-21
parser.add_argument("--num_sample",type=int,default=2000,help="the number of selected background samples in the HSI")
parser.add_argument("--ratio",type=int,default=0.2,help="the ratio of remove residual targets")#ratio是一个百分比
args = parser.parse_args()

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


def get_back_samples(prior_target,image,args):
    # Selecting background samples
    selected_back = selec_back(prior_target, image, num_back=args.num_sample, ratio=args.ratio)
    selected_back = torch.Tensor(selected_back)
    selected_back = selected_back.cpu().numpy()

    return selected_back


def selec_back(prior_target,image,num_back,ratio):

    """ Selection of background samples using peak density clustering """
    
    sim = F.cosine_similarity(prior_target,image,dim=1)
    sorted_sim,indices = torch.sort(sim,descending=False)
    selec_num = torch.ceil(torch.mul(1-ratio,sorted_sim.shape[0])).to(dtype=int)
    indices = indices.cpu().numpy()
    selec_idx = indices[0:selec_num]
    rho_threshold = 50
    for i in selec_idx:
        x0 = torch.unsqueeze(image[i],dim=0)
        rho,_ = clustering(x0,image,num_back)
        if rho[0] >= rho_threshold:#
            main_back = x0
            break

    for i in selec_idx:
        x = torch.unsqueeze(image[i],dim=0)
        value = F.cosine_similarity(x,main_back)
        one = torch.ones_like(value)
        zero = torch.zeros_like(value)
        value = torch.where(value<0.899,zero,one)
    
        if torch.sum(value) == 0:
            main_back = torch.cat([main_back,x],dim=0)

    _,selected_back = clustering(main_back,image,num_back)
    print(selected_back.shape)
    return selected_back


def clustering(x,image,num_back):
    N = x.shape[0]
    print(N)
    rho = []
    indicesBacks = []
    delta_threshold = 0.9

    for i in range(N):
        dist = F.cosine_similarity(torch.unsqueeze(x[i],dim=0),image)
        sorted_dist,indices = torch.sort(dist,descending=True)
        sorted_dist = sorted_dist.tolist()
        num = 0
        while(True):
            if(sorted_dist[num] < delta_threshold):
                break
            else:
                num += 1
        rho.append(num)
        print(rho)
        indicesBacks.append(indices[0:num])

    avg_n = torch.floor(torch.tensor(num_back/N)).type(torch.int)
    avg_n = avg_n.item()
    indices_back = []
    for i in range(N):
        r = rho[i]
        if r >= avg_n:
            indices_back.append(indicesBacks[i][0:avg_n])
        else:
            indices_back.append(indicesBacks[i][0:r])

    B = indices_back[0]
    if N > 1:
        for i in range(N):
            id_back = indices_back[i]
            B = torch.cat([B,id_back],dim=0)
    selected_back = []
    for i in B.cpu().numpy():
        selected_back.append(image[i])
    selected_back = torch.stack(selected_back,dim=0)

    return rho,selected_back


if __name__=='__main__':
    prior_target, image, mask = data_reader(
        data_path=os.path.join(args.data_dir, '{}.mat'.format(args.data_name)), device=used_device)
    height, width = mask.shape
    selected_back = get_back_samples(prior_target, image, args)
    #保存到指定文件
    # tensor_cpu = selected_back.cpu()  # 将张量复制到主机内存中
    # selected_back = tensor_cpu.detach().numpy()  # 将张量转换为NumPy数组
    sio.savemat('./data/AV_selected_back.mat', {'selected_back': selected_back})



