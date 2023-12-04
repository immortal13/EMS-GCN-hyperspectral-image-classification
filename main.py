import numpy as np
import scipy.io as sio
import random
import time,math
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import EMSGCN
from utils import evaluate_performance, GT_To_One_Hot, compute_loss

import argparse
parser = argparse.ArgumentParser(description='args')
parser.add_argument('--data', type=int, default=2) #dataset
parser.add_argument('--tr', type=float, default=30) #train_samples_per_class
parser.add_argument('--scale', type=int, default = 100) #superpixel segmenation scale 
parser.add_argument('--sample', type=int, default = 0) #sample strategy, 0 is random sample, 1 is disjoint sample
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CompactnessLoss(nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, posclr_s, posclr):
        return super().forward(posclr_s[:,:2], posclr[:,:2].detach())
compact_loss =  CompactnessLoss()

for (FLAG, train_samples_per_class,superpixel_scale) in [(args.data,args.tr,args.scale)]:
    torch.cuda.empty_cache()

    Train_Time_ALL=[]
    Test_Time_ALL=[]
    
    Seed_List=[i for i in range(5)]
    if FLAG == 1:
        data_mat = sio.loadmat('/home/zoujiaqi/Mix-GCNet/HyperImage_data/indian/Indian_pines_corrected.mat')
        data = data_mat['indian_pines_corrected']
        gt_mat = sio.loadmat('/home/zoujiaqi/Mix-GCNet/HyperImage_data/indian/Indian_pines_gt.mat')
        gt = gt_mat['indian_pines_gt']

        val_ratio = 0.01  
        class_count = 16  
        learning_rate = 5e-4 
        max_epoch = 600 
        dataset_name = "indian_"

    if FLAG == 2:
        data_mat = sio.loadmat('/home/zoujiaqi/Mix-GCNet/HyperImage_data/paviaU/PaviaU.mat')
        data = data_mat['paviaU']
        gt_mat = sio.loadmat('/home/zoujiaqi/Mix-GCNet/HyperImage_data/paviaU/PaviaU_gt.mat')
        gt = gt_mat['paviaU_gt']

        class_count = 9 
        learning_rate = 5e-4 
        max_epoch = 300 
        dataset_name = "paviaU_"
        
    # positional pixel features I^xy
    height, width, bands = m, n, d = data.shape 
    coords = torch.stack(torch.meshgrid(torch.arange(m, device=device), torch.arange(n, device=device)), 0)
    coords = coords[None].repeat(1, 1, 1, 1).float()

    # normalization of HSI data X
    orig_data=data
    data = np.reshape(data, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)
    data = np.reshape(data, [height, width, bands])
    print()
    print("❀"*20)
    print("Step1: load HSI dataset.")
    print("data:",dataset_name, "train_samples_per_class:", train_samples_per_class,"superpixel_scale:", superpixel_scale)
    print("❀"*20)
    print()

    # repeat 5 times to calculate the average/std value
    for curr_seed in Seed_List:
        # set seed to generate training samples for each time
        random.seed(curr_seed)

        gt_1d = np.reshape(gt, [-1])
        train_rand_idx = []
        
        # sample selection strategy 1: randomly select L training samples for each class
        if args.sample == 0:
            for i in range(class_count):
                idx = np.where(gt_1d == i + 1)[-1]
                samplesCount = len(idx)
                real_train_samples_per_class = train_samples_per_class
                rand_list = [i for i in range(samplesCount)]  
                if real_train_samples_per_class > samplesCount:
                    real_train_samples_per_class = real_train_samples_per_class//2 # e.g. For Indian Pines, some categories < 30 labels, then choose 15 samples
                rand_idx = random.sample(rand_list,
                                         real_train_samples_per_class)
                rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
                train_rand_idx.append(rand_real_idx_per_class_train)

            ## aggregate all classes
            train_rand_idx = np.array(train_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)
            
            ## prepare training map and test map , their shape are ([height,width])
            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_1d))]
            all_data_index = set(all_data_index)

            background_idx = np.where(gt_1d == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index - background_idx

            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)

            train_samples_gt = np.zeros(gt_1d.shape)
            for i in range(len(train_data_index)):
                train_samples_gt[train_data_index[i]] = gt_1d[train_data_index[i]]

            test_samples_gt = np.zeros(gt_1d.shape)
            for i in range(len(test_data_index)):
                test_samples_gt[test_data_index[i]] = gt_1d[test_data_index[i]]
            Test_GT = np.reshape(test_samples_gt, [m, n]) 

            train_samples_gt=np.reshape(train_samples_gt,[height,width])
            test_samples_gt=np.reshape(test_samples_gt,[height,width])
            num_train = len(train_data_index)
            num_test = len(test_data_index)

        ## prepare onehot matrix, their shape are ([m * n, class_count])
        train_samples_gt_onehot=GT_To_One_Hot(train_samples_gt,class_count)
        test_samples_gt_onehot=GT_To_One_Hot(test_samples_gt,class_count)
        train_samples_gt_onehot=np.reshape(train_samples_gt_onehot,[-1,class_count]).astype(int)
        test_samples_gt_onehot=np.reshape(test_samples_gt_onehot,[-1,class_count]).astype(int)

        ## prepare training mask and test mask , their shape are ([m * n, class_count])
        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m * n, class_count])

        test_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m * n, class_count])

        # cpu -> gpu
        train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
        test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
        train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
        test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
        train_samples_gt_onehot_=train_samples_gt_onehot.reshape(height, width,-1).permute(2,0,1).unsqueeze(0)
        train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
      
        net_input=np.array(data,np.float32)
        net_input=torch.from_numpy(net_input.astype(np.float32)).to(device)

        net = EMSGCN.EMSGCN(height, width, bands, class_count, superpixel_scale)
        net.to(device)

        print("❀"*20)
        print("Step2: prepare input data, training map(label), test map(label).")
        print("num of training/test label: ",num_train," / ",num_test)
        print("❀"*20)
        print()

        # training stage
        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)#,weight_decay=0.0001
        net.train()
        tic1 = time.time()
        for i in range(max_epoch+1):
            optimizer.zero_grad()             
            output, Q, ops, H = net(net_input)
            ## CE loss
            loss = compute_loss(output,train_samples_gt_onehot,train_label_mask)
            ## reconstruction loss        
            R = train_samples_gt_onehot_
            spf_R = ops['map_p2sp'](R.float().contiguous(), Q)            
            R_recon = ops['map_sp2p'](spf_R, Q)
            R_recon_label = R_recon[0].permute(1,2,0).reshape(-1,output.shape[-1]).float()
            R_recon_label = F.softmax(R_recon_label, -1)            
            recon_loss = compute_loss(R_recon_label, train_samples_gt_onehot, train_label_mask)#reconstruct_loss(R_recon, R) #torch.Size([1, 16, 145, 145])
            ## compact loss  
            spf_x = ops['map_p2sp'](coords, Q)
            spixel_map = ops['smear'](spf_x, H.detach())
            cpt_loss = compact_loss(spixel_map, coords) #torch.Size([1, 2, 145, 145])
            lamada1 = 0.1
            lamada2 = 0.001
            loss = loss+lamada1*recon_loss+lamada2*cpt_loss
            
            loss.backward(retain_graph=False)
            optimizer.step()  

            # print loss/acc
            if i%100==0:
                with torch.no_grad():
                    net.eval()
                    output, _, _, H = net(net_input)
                    ## CE loss
                    trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
                    trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot, m, n, class_count, Test_GT)
                    testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
                    testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot, m, n, class_count, Test_GT)
                    print("{}\ttrain loss={}\t train OA={} test loss={}\t test OA={}".format(str(i + 1), round(trainloss.item(),5), round(trainOA.item(),5), round(testloss.item(),5), round(testOA.item(),5)))
                    torch.save(net.state_dict(),"model/{}best_model.pt".format(dataset_name))
                torch.cuda.empty_cache()
                net.train()
        toc1 = time.time()
        print("\n\n====================training done. starting evaluation...========================\n")
        training_time=toc1 - tic1 #分割耗时需要算进去
        Train_Time_ALL.append(training_time)
        
        # test stage
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load("model/{}best_model.pt".format(dataset_name)))
            net.eval()
            tic2 = time.time()
            output, _, _,  H = net(net_input)
            toc2 = time.time()
            testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
            testOA, OA_ALL, AA_ALL, KPP_ALL, AVG_ALL = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot, m, n, class_count, Test_GT, require_AA_KPP=True, printFlag=False)
            print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))
            
            ## save classification map
            classification_map=torch.argmax(output, 1).reshape([height,width]).cpu()+1
            pred_mat=classification_map.data.numpy()
            sio.savemat("{}_EMSGCN_pred_mat.mat".format(dataset_name),{"pred_mat":pred_mat})
            testing_time=toc2 - tic2 #分割耗时需要算进去
            Test_Time_ALL.append(testing_time)
        torch.cuda.empty_cache()
        del net
        
    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    Train_Time_ALL=np.array(Train_Time_ALL)
    Test_Time_ALL=np.array(Test_Time_ALL)
    
    print("❀"*20)
    print("Step3: training and test.")
    print("❀"*20)
    print()

    print("\ntrain_ratio={}".format(train_samples_per_class),
          "\n==============================================================================")
    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))
