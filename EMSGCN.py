import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from ssn import SSN
from utils import FeatureConverter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SEBlock_diag(nn.Module):
    def __init__(self, in_channels):
        super(SEBlock_diag, self).__init__()
        reduction_ratio=16
        self.gap1 = nn.AdaptiveAvgPool1d(1) ##B C L→B C 1
        self.seq = nn.Sequential(
            nn.Linear(in_channels, in_channels), 
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x): # x.shape=[L,C]
        x=torch.unsqueeze(x.t(), 0)
        v = self.gap1(x)
        v=v.reshape(1,-1)
        value1 = self.seq(v).reshape(-1)
        diag=torch.diag(self.sigmoid(value1))
        return diag #(C,C)

class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(GCNLayer, self).__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.GCN_linear_theta_1 =nn.Sequential(nn.Linear(input_dim, 256))

        self.GCN_liner_out_1 =nn.Sequential(nn.Linear(input_dim, output_dim//2))
        self.GCN_liner_out_2 =nn.Sequential(nn.Linear(input_dim, output_dim//2))

    def forward(self, H, A):
        nodes_count=A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        self.mask=torch.ceil(A*0.01)#+self.I 

        H = self.BN(H)
        H_xx1= self.GCN_linear_theta_1(H)

        e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        zero_vec = -9e15 * torch.ones_like(e)
        A = torch.where(self.mask > 0, e, zero_vec) #+ self.I 
        A.fill_diagonal_(1) 
        A = F.softmax(A, dim=1)

        output1 = self.Activition(torch.mm(self.I, self.GCN_liner_out_1(H)))
        output2 = self.Activition(torch.mm(A, self.GCN_liner_out_2(H)))
        output = torch.cat([output1,output2],dim=-1)

        return output

class EMSGCN(nn.Module):
    def __init__(self, height: int, width: int, channel: int, class_count: int, superpixel_scale):
        super(EMSGCN, self).__init__()
        # 类别数,即网络最终输出通道数
        self.Activition = nn.LeakyReLU()
        self.class_count = class_count  # 类别数
        self.channel = channel #输入通道数
        self.height = height
        self.width = width  
        
        self.n_spixels = n_spixels = int(self.height*self.width/superpixel_scale)    #超像素个数 
        n_iters = 5  # iteration of DSLIC
        n_filters = 128 # feature channel of CNN in SSN module 
        ETA_POS = 1.8 # scale coefficient of positional pixel features I^xy, it is very important
        
        GCN_hidden = [128,128]
        global A # adjacency matrix
        A=0

        self.ssn = SSN(FeatureConverter(ETA_POS), n_iters, n_spixels, n_filters, channel, cnn=True)
        self.gcn = nn.Sequential(
                GCNLayer(GCN_hidden[0], GCN_hidden[0]),
                GCNLayer(GCN_hidden[0], GCN_hidden[1]),
                )
        self.se_1=SEBlock_diag(GCN_hidden[1]) # channel attention module
        self.se_2=SEBlock_diag(GCN_hidden[1]) # channel attention module
        self.fc = nn.Sequential(nn.Linear(GCN_hidden[1], self.class_count))

    def get_A(self,segments_map):
        # print(np.max(segments_map),self.n_spixels)
        A = np.zeros([self.n_spixels,self.n_spixels], dtype=np.float32)
        (h, w) = segments_map.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = segments_map[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue
                    A[idx1, idx2] = A[idx2, idx1] = 1                
        A=torch.from_numpy(A).cuda()
        return A

    def forward(self, x: torch.Tensor):        
        (h, w, c) = x.shape
        x_cnn = torch.unsqueeze(x.permute([2, 0, 1]), 0) #(1,C,H,W)

        ## differentiable SLIC (DSLIC)
        # Q:torch.Size([1, 9, 145, 145]) f:torch.Size([1, 202, 145, 145]) spf:torch.Size([1, 128, 841]) pf:torch.Size([1, 128, 145, 145])
        # Q is the pixel-superpixel assignment matrix, f = pos + HSI, spf is superpixel feature, pf is pixel feature
        Q, ops, f, spf, pf = self.ssn(x_cnn) 
        Q_d = Q.detach()

        ########### GCN branch ########### 
        segments_map_cuda = ops['map_idx'](torch.argmax(Q_d, 1, True).int()) 
        segments_map = segments_map_cuda[0,0].cpu().numpy() #[1, 1, 145, 145]
        self.n_spixels = np.max(segments_map) + 1
        global A
        if isinstance(A,int):
            A=self.get_A(segments_map)

        gcn_feat = self.gcn[0](spf[0].t(), A)
        diag = self.se_1(gcn_feat)
        gcn_feat=torch.matmul(gcn_feat, diag)
        
        gcn_feat = self.gcn[1](gcn_feat, A)
        diag = self.se_2(gcn_feat)
        gcn_feat=torch.matmul(gcn_feat, diag)
        gcn_result = ops['map_sp2p'](torch.unsqueeze(gcn_feat.t(),0).contiguous(), Q_d)

        ########### Softmax classifier ########### 
        Y = gcn_result
        Y = torch.squeeze(Y, 0).permute([1, 2, 0]).reshape([h * w, -1])
        Y = self.fc(Y) #(h * w, #classes)
        Y = F.softmax(Y, -1)
        return Y, Q, ops, segments_map_cuda


