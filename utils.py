import math
import torch
import numpy as np
from sklearn import metrics
from scipy.interpolate import RegularGridInterpolator
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######### related to ssn.py #########
def init_grid(n_spixels_expc, w, h):
    # n_spixels >= n_spixels_expc
    nw_spixels = math.ceil(math.sqrt(w*n_spixels_expc/h))
    nh_spixels = math.ceil(math.sqrt(h*n_spixels_expc/w))

    n_spixels = nw_spixels*nh_spixels   # Actual number of spixels

    if n_spixels > w*h:
        raise ValueError("Superpixels must be fewer than pixels!")
        
    w_spixel, h_spixel = (w+nw_spixels-1) // nw_spixels, (h+nh_spixels-1) // nh_spixels
    rw, rh = w_spixel*nw_spixels-w, h_spixel*nh_spixels-h

    if (rh/2 + h_spixel) < 0 or (rw/2 + w_spixel) < 0 or (rh/2-h_spixel) > 0 or (rw/2-w_spixel) > 0:
        raise ValueError("The expected number of superpixels does not fit the image size!")

    y = np.array([-1, *np.arange((h_spixel-1)/2, h+rh, h_spixel), h+rh])-rh/2
    x = np.array([-1, *np.arange((w_spixel-1)/2, w+rw, w_spixel), w+rw])-rw/2

    s = np.arange(n_spixels).reshape(nh_spixels, nw_spixels).astype(np.int32)
    s = np.pad(s, ((1,1),(1,1)), 'edge')
    f = RegularGridInterpolator((y, x), s, method='nearest')

    pts = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    pts = np.stack(pts, axis=-1)
    init_idx_map = f(pts).astype(np.int32)
    
    return init_idx_map, n_spixels, nw_spixels, nh_spixels

######### related to EMSGCN.py #########
class FeatureConverter:
    def __init__(self, eta_pos=2, gamma_clr=0.1):
        super().__init__()
        self.eta_pos = eta_pos
        self.gamma_clr = gamma_clr

    def __call__(self, feats, nw_spixels, nh_spixels):
        # Do not require grad
        b, c, h, w = feats.size()

        pos_scale = self.eta_pos*max(nw_spixels/w, nh_spixels/h)   
        coords = torch.stack(torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device)), 0)
        coords = coords[None].repeat(feats.shape[0], 1, 1, 1).float()
        # print(pos_scale)
        feats = torch.cat([feats, pos_scale*coords], 1)#(1,202,145,145)
        # feats.requires_grad = True
        return feats

######### related to Main.py #########
OA_ALL = []
AA_ALL = []
KPP_ALL = []
AVG_ALL = []

def evaluate_performance(network_output,train_samples_gt,train_samples_gt_onehot, m, n, class_count, Test_GT, require_AA_KPP=False,printFlag=True):
    zeros = torch.zeros([m * n]).to(device).float()
    if False==require_AA_KPP:
        with torch.no_grad():
            available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
            available_label_count=available_label_idx.sum()#有效标签的个数
            correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
            OA= correct_prediction.cpu()/available_label_count
            
            return OA
    else:
        with torch.no_grad():
            #计算OA
            available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
            available_label_count=available_label_idx.sum()#有效标签的个数
            correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
            OA= correct_prediction.cpu()/available_label_count
            OA=OA.cpu().numpy()
            
            # 计算AA
            zero_vector = np.zeros([class_count])
            output_data=network_output.cpu().numpy()
            train_samples_gt=train_samples_gt.cpu().numpy()
            train_samples_gt_onehot=train_samples_gt_onehot.cpu().numpy()
            
            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)
            for z in range(output_data.shape[0]):
                if ~(zero_vector == output_data[z]).all():
                    idx[z] += 1
            
            count_perclass = np.zeros([class_count])
            correct_perclass = np.zeros([class_count])
            for x in range(len(train_samples_gt)):
                if train_samples_gt[x] != 0:
                    count_perclass[int(train_samples_gt[x] - 1)] += 1
                    if train_samples_gt[x] == idx[x]:
                        correct_perclass[int(train_samples_gt[x] - 1)] += 1
            test_AC_list = correct_perclass / count_perclass
            test_AA = np.average(test_AC_list)

            # 计算KPP
            test_pre_label_list = []
            test_real_label_list = []
            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)
            idx = np.reshape(idx, [m, n])
            for ii in range(m):
                for jj in range(n):
                    if Test_GT[ii][jj] != 0:
                        test_pre_label_list.append(idx[ii][jj] + 1)
                        test_real_label_list.append(Test_GT[ii][jj])
            test_pre_label_list = np.array(test_pre_label_list)
            test_real_label_list = np.array(test_real_label_list)
            kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                              test_real_label_list.astype(np.int16))
            test_kpp = kappa

            # 输出
            if printFlag:
                print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                print('acc per class:')
                print(test_AC_list)

            OA_ALL.append(OA)
            AA_ALL.append(test_AA)
            KPP_ALL.append(test_kpp)
            AVG_ALL.append(test_AC_list)
           
            return OA,OA_ALL,AA_ALL,KPP_ALL,AVG_ALL

def GT_To_One_Hot(gt, class_count):
        '''
        Convet Gt to one-hot labels
        :param gt:
        :param class_count:
        :return:
        '''
        GT_One_Hot = []  # 转化为one-hot形式的标签
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                temp = np.zeros(class_count,dtype=np.float32)
                if gt[i, j] != 0:
                    temp[int(gt[i, j]) - 1] = 1
                GT_One_Hot.append(temp)
        GT_One_Hot = np.reshape(GT_One_Hot, [gt.shape[0], gt.shape[1], class_count])
        return GT_One_Hot

def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
            real_labels = reallabel_onehot
            we = -torch.mul(real_labels,torch.log(predict))
            we = torch.mul(we, reallabel_mask)
            pool_cross_entropy = torch.sum(we)#/270#/float(len(train_data_index))
            return pool_cross_entropy