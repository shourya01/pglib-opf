import os, pickle
import numpy as np
import torch.nn.functional as F
import torch
import cyipopt
import warnings
from tqdm import trange
from pypower.ext2int import ext2int
from pypower.api import loadcase
from problemDefJITM import opfSocp
from problemDefJITMargin import opfSocpMargin
from utils import make_data_parallel
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import gc
from time import time
from oct2py import Oct2Py
from ConvexModel import ConvexNet
from ICGN import ICGN
from ClassifierModel import ClassifierNet
from RidgeModel import RidgeNet
import xgboost as xgb
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

octave = Oct2Py()
dir_name = 'data/'
MAX_BUS = 10000
DUAL_THRES = 1e-4
DEVICE_MODULO = 2

if __name__ == "__main__":
    
    # print number of devices
    print(f"Detected {torch.cuda.device_count()} gpus!")
    
    # # get all cases in current directory
    # current_directory = os.getcwd()+'/'
    # # current_directory = '/home/sbose/pglib-opf/' # for running on BEBOP
    # all_files_and_directories = os.listdir(current_directory)
    # # three specific cases
    # todo_cases =  ['pglib_opf_case118_ieee','pglib_opf_case2312_goc','pglib_opf_case4601_goc','pglib_opf_case10000_goc']
    
    # with open(os.getcwd()+'/allcases.pkl','rb') as file:
    #     data = pickle.load(file)
        
    # cases = data['cases']
    # casenames = data['casenames']
    # cases_new, casenames_new = [], []
    # for cs,cn in zip(cases,casenames):
    #     if cn in todo_cases:
    #         cases_new.append(cs)
    #         casenames_new.append(cn)
    # cases, casenames = cases_new, casenames_new
    
    # get all cases in current directory
    current_directory = os.getcwd()+'/'
    # current_directory = '/home/sbose/pglib-opf/' # for running on BEBOP
    all_files_and_directories = os.listdir(current_directory)
    # three specific cases
    case_files = [current_directory+i for i in ['pglib_opf_case118_ieee.m','pglib_opf_case793_goc.m','pglib_opf_case1354_pegase.m','pglib_opf_case2312_goc.m','pglib_opf_case4601_goc.m','pglib_opf_case10000_goc.m']]
    # case_files = [current_directory+i for i in ['pglib_opf_case2312_goc.m',"pglib_opf_case4601_goc.m"]]

    cases, casenames = [], []
    cases_full, casenames_full = [], []
    
    for cf in case_files:
        octave.source(current_directory+os.path.basename(cf))
        cname = os.path.basename(cf).split('.')[0]
        num_buses = None
        # determine number of buses in the case from its name
        for ci in cname.split('_'):
            if 'case' in ci:
                num_buses = int(''.join([chr for chr in ci.replace('case','',1) if chr.isdigit()]))
        # fitler out cases with more buses than MAX_BUS
        if num_buses <= MAX_BUS:
            # convert to internal indexing
            case_correct_idx = ext2int(loadcase(octave.feval(cname)))
            # append
            cases.append(case_correct_idx)
            casenames.append(cname)
    
    # write
    logfile = open('perf.txt','w')
            
    for cn,this_case in zip(casenames,cases):
        
        print(f"Solving case {cn}.",flush=True)
        
        # torch.set_default_dtype(torch.bfloat16)
    
        inp_data = np.load(os.getcwd()+'/'+dir_name+f'{cn}_inp.npz')['data']
        dual_data = np.load(os.getcwd()+'/'+dir_name+f'{cn}_dual.npz')['data']
        # cost_data = np.load(os.getcwd()+'/'+dir_name+f'{cn}_costs.npz')['data'][:,None]
        
        # print(f"Raw input sizes are {inp_data.shape}, raw dual sizes are {dual_data.shape}.")
        
        # save test inputs
        np.savez_compressed(os.getcwd()+f'/saved/{cn}_test_inp.npz',data=inp_data)
        
        # modify the input
        optObj = opfSocp(this_case,cn) # generate object
        
        # partition the data into train and test
        inp_data_train = inp_data[:int(0.8*inp_data.shape[0]),:]
        print(inp_data_train.shape,flush=True)
        
        # acquire and preprocess train data
        inp_min, inp_max = inp_data_train.min(), inp_data_train.max() # preprocess - acquire min and max
        inp_data_train = (inp_data_train - inp_min) / (inp_max - inp_min) # preprocess - min-max scaling
        # cost_data_train = cost_data[:int(0.8*inp_data.shape[0]),:]
        dual_data_train = dual_data[:int(0.8*inp_data.shape[0]),:]
        dual_data_train = dual_data_train.copy() # save copy for later
        
        # importance weights
        dual_train_bind = np.where(np.abs(dual_data_train)>DUAL_THRES,1,0)
        dual_train_nobind = np.where(np.abs(dual_data_train)<DUAL_THRES,1,0)
        WEIGHT_FOR_BINDING_CONSTR = int(10000*min(optObj.n_bus,1000))
        
        # acquire and preprocess test data
        inp_data_test = inp_data[int(0.8*inp_data.shape[0]):,:]
        inp_data_test_unscaled = inp_data_test.copy()
        inp_data_test = (inp_data_test - inp_min) / (inp_max - inp_min) # preprocess - min-max scaling
        # cost_data_test = cost_data[int(0.8*inp_data.shape[0]):,:]
        dual_data_test = dual_data[int(0.8*inp_data.shape[0]):,:]
        dual_data_test = np.where(np.abs(dual_data_test)<DUAL_THRES,0.,1.) # preprocess to easily calculate confusion matrix
        best_test = np.zeros_like(dual_data_test)
        
        # vector for nonmodel inequalities
        nmineq = np.ones(2*optObj.n_bus+4*optObj.n_branch+2*optObj.in_size)
        nmineq[:2*optObj.n_bus] = 0
        nmineq = nmineq.astype(bool)
        
        # save dual data
        np.savez_compressed(os.getcwd()+f'/saved/{cn}_test_gt.npz',data=dual_data_test[:,nmineq])
        
        # create convex net
        nn_shape = (inp_data.shape[1],300,150,150,1)
        # cModel = ConvexNet(nn_shape,activation=nn.LeakyReLU).to('cuda')
        cModel = ICGN(in_dim=nn_shape[0],hidden=1).to('cuda')
        # cModel = nn.Sequential(
        #     nn.Linear(nn_shape[0],1000),
        #     nn.LeakyReLU(),
        #     nn.Linear(1000,500),
        #     nn.LeakyReLU(),
        #     nn.Linear(500,500),
        #     nn.LeakyReLU(),
        #     nn.Linear(500,nn_shape[0])
        # ).to('cuda')
        for p in cModel.parameters():
            p.data.fill_(0.01)
        BS = 32 # batch size
        optimConvex = torch.optim.Adam(cModel.parameters(),lr=1e-4,weight_decay=0)
        epochs = 2000
        losses = []
        FP, FN, TP, TN = [], [], [], []
        FP_best, FN_best, TP_best, TN_best = 0, dual_data_test.size, 0, 0
        times = 0
        
        # carry out training for convex neural net
        first_test_done = False
        best_recorded = False
        best_epoch = 0
        start = time()
        for e in range(epochs):
            np.random.seed(e)
            random_idx = np.random.choice(inp_data_train.shape[0],BS,replace=False)
            inp_t = torch.FloatTensor(inp_data_train[random_idx,:]).to(torch.float32).to('cuda:0')
            grad_t = torch.FloatTensor(dual_data_train[random_idx,:]).to(torch.float32).to('cuda:0')
            wt = torch.FloatTensor(np.where(np.abs(dual_data_train[random_idx,:])<DUAL_THRES,1,WEIGHT_FOR_BINDING_CONSTR)).to(torch.float32).to('cuda:0')
            # # for MONOTONIC
            # loss_grad = F.binary_cross_entropy_with_logits(cModel(inp_t,mode='train_monotone'),torch.sigmoid(grad_t),weight=wt)
            # optimConvex.zero_grad()
            # loss_grad.backward()
            # optimConvex.step()
            # # for MONOTONE
            # loss_grad = F.binary_cross_entropy_with_logits(cModel(inp_t,mode='train_monotone'),torch.sigmoid(grad_t),weight=wt)
            # optimConvex.zero_grad()
            # loss_grad.backward()
            # optimConvex.step()
            # # for ROUTER
            # loss_grad = F.binary_cross_entropy_with_logits(cModel(inp_t,mode='train_router'),torch.sigmoid(grad_t),weight=wt)
            # optimConvex.zero_grad()
            # loss_grad.backward()
            # optimConvex.step()
            # # FCNN
            # loss_grad = F.binary_cross_entropy_with_logits(cModel(inp_t),torch.sigmoid(grad_t),weight=wt)
            # optimConvex.zero_grad()
            # loss_grad.backward()
            # optimConvex.step()
            # ICGN
            loss_grad = F.binary_cross_entropy_with_logits(cModel(inp_t,N=100),torch.sigmoid(grad_t),weight=wt)
            optimConvex.zero_grad()
            loss_grad.backward()
            optimConvex.step()
            del wt
            del grad_t
            losses.append(loss_grad.item())
            # test
            if (e+1) % 250 == 0:
                # train the router
                # for _ in range(250):
                #     random_idx = np.random.choice(inp_data_train.shape[0],BS,replace=False)
                #     inp_t = torch.FloatTensor(inp_data_train[random_idx,:]).to(torch.float32).to('cuda:0')
                #     grad_t = torch.FloatTensor(dual_data_train[random_idx,:]).to(torch.float32).to('cuda:0')
                #     wt = torch.FloatTensor(np.where(np.abs(dual_data_train[random_idx,:])<DUAL_THRES,1,WEIGHT_FOR_BINDING_CONSTR)).to(torch.float32).to('cuda:0')
                #     loss_grad = F.binary_cross_entropy_with_logits(cModel(inp_t,mode='train_router'),torch.sigmoid(grad_t),weight=wt)
                #     optimConvex.zero_grad()
                #     loss_grad.backward()
                #     optimConvex.step()
                inp = torch.FloatTensor(inp_data_test).to(torch.float32).to('cuda')
                out_test = cModel(inp,N=100).detach().cpu().numpy()
                # out_test = cModel(inp).detach().cpu().numpy()
                del inp
                out_test = np.where(np.abs(out_test)<DUAL_THRES,0,1)
                tp, tn, fp, fn = out_test[:,nmineq]*dual_data_test[:,nmineq],\
                    (1-out_test[:,nmineq])*(1-dual_data_test[:,nmineq]),\
                    out_test[:,nmineq]*(1-dual_data_test[:,nmineq]),\
                    (1-out_test[:,nmineq])*(dual_data_test[:,nmineq])
                tp, tn, fp, fn = tp.sum().item(), tn.sum().item(), fp.sum().item(), fn.sum().item()
                FP.append(fp)
                TP.append(tp)
                FN.append(fn)
                TN.append(tn) 
                first_test_done = True
                # print
                if first_test_done:
                    allv = tp + tn + fp + fn
                    print(f"For case {cn}, method: convex, loss on epoch {e+1} is {losses[-1]:.5f}. TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}.",flush=True)
                    if fn <= FN_best and (tn*100/allv) > 50: # best false negative when true negatives is above 50 percent
                        best_recorded, best_epoch = True, e+1
                        TP_best = tp
                        TN_best = tn
                        FP_best = fp
                        FN_best = fn
                        best_test = out_test
                        torch.save(cModel.state_dict(),os.getcwd()+f'/saved/{cn}_ConvexModel.pth')
                else:
                    print(f"For case {cn}, method: convex, loss on epoch {e+1} is {losses[-1]:.5f}.",flush=True)
                # log
                if (e+1) == epochs:
                    if best_recorded:
                        times = time() - start
                        logfile.write(f"For case {cn}, method: convex, best performance recorded on epoch {best_epoch}/{epochs}.\nTP: {TP_best*100/allv:.5f}, TN: {TN_best*100/allv:.5f}, FP: {FP_best*100/allv:.5f}, FN: {FN_best*100/allv:.5f}, train_time: {times}s.\n")
                    else:
                        logfile.write(f"For case {cn}, method: convex. TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}.\n")
        if not best_recorded:
            torch.save(cModel.state_dict(),os.getcwd()+f'/saved/{cn}_ConvexModel.pth')
        if best_recorded:
            np.savez_compressed(os.getcwd()+f'/saved/{cn}_out_convex.npz',data=best_test)
        else:
            np.savez_compressed(os.getcwd()+f'/saved/{cn}_out_convex.npz',data=out_test)
        del cModel
        gc.collect()
        torch.cuda.empty_cache()
                
        # # now create classifier
        # dual_data_train_classifier = np.where(np.abs(dual_data_train)<DUAL_THRES,0,1)
        # ClassifierModel = make_data_parallel(ClassifierNet(nn_shape,activation=nn.LeakyReLU).to('cuda'),optObj.n_bus)
        # for p in ClassifierModel.parameters():
        #     p.data.fill_(0.01)
        # optimClassifier = torch.optim.Adam(ClassifierModel.parameters(),lr=1e-4)
        # losses = []
        # FP, FN, TP, TN = [], [], [], []
        # FP_best, FN_best, TP_best, TN_best = 0, dual_data_test.size, 0, 0
        # times = 0
        
        # # carry out training for classifier neural net
        # first_test_done = False
        # best_recorded = False
        # best_epoch = 0
        # start = time()
        # for e in range(epochs):
        #     np.random.seed(e)
        #     random_idx = np.random.choice(inp_data_train.shape[0],BS,replace=False)
        #     inp_t = torch.FloatTensor(inp_data_train[random_idx,:]).to(torch.float32).to('cuda')
        #     probs_t = torch.FloatTensor(dual_data_train_classifier[random_idx,:]).to(torch.float32).to('cuda')
        #     loss_grad = F.cross_entropy(ClassifierModel(inp_t),probs_t,reduction='mean')
        #     # loss = loss_grad
        #     optimClassifier.zero_grad()
        #     loss_grad.backward()
        #     optimClassifier.step()
        #     del inp_t
        #     del probs_t
        #     losses.append(loss_grad.item())
        #     # test
        #     if (e+1) % 250 == 0:
        #         inp = torch.FloatTensor(inp_data_test).to(torch.float32).to('cuda')
        #         out_test = ClassifierModel(inp).detach().cpu().numpy()
        #         del inp
        #         out_test = np.where(out_test<0.5,0,1)
        #         tp, tn, fp, fn = out_test[:,nmineq]*dual_data_test[:,nmineq],\
        #             (1-out_test[:,nmineq])*(1-dual_data_test[:,nmineq]),\
        #             out_test[:,nmineq]*(1-dual_data_test[:,nmineq]),\
        #             (1-out_test[:,nmineq])*(dual_data_test[:,nmineq])
        #         tp, tn, fp, fn = tp.sum().item(), tn.sum().item(), fp.sum().item(), fn.sum().item()
        #         FP.append(fp)
        #         TP.append(tp)
        #         FN.append(fn)
        #         TN.append(tn) 
        #         first_test_done = True
        #         # print
        #         if first_test_done:
        #             allv = tp + tn + fp + fn
        #             print(f"For case {cn}, method: classifier, loss on epoch {e+1} is {losses[-1]:.5f}. TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}s.",flush=True)
        #             if fn <= FN_best and (tn*100/allv) > 50: # best false negative when true negatives fall below 50 percent
        #                 best_recorded, best_epoch = True, e+1
        #                 TP_best = tp
        #                 TN_best = tn
        #                 FP_best = fp
        #                 FN_best = fn
        #                 best_test = out_test
        #                 torch.save(ClassifierModel.state_dict(),os.getcwd()+f'/saved/{cn}_ClassifierModel.pth')
        #         else:
        #             print(f"For case {cn}, method: classifier, loss on epoch {e+1} is {losses[-1]:.5f}.",flush=True)
        #         # log
        #         if (e+1) == epochs:
        #             if best_recorded:
        #                 times = time() - start
        #                 logfile.write(f"For case {cn}, method: classifier, best performance recorded on epoch {best_epoch}/{epochs}.\nTP: {TP_best*100/allv:.5f}, TN: {TN_best*100/allv:.5f}, FP: {FP_best*100/allv:.5f}, FN: {FN_best*100/allv:.5f}, train_time: {times}s.\n")
        #             else:
        #                 logfile.write(f"For case {cn}, method: classifier. TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}.\n")
        # if not best_recorded:
        #     torch.save(ClassifierModel.state_dict(),os.getcwd()+f'/saved/{cn}_ClassifierModel.pth')
        # if best_recorded:
        #     np.savez_compressed(os.getcwd()+f'/saved/{cn}_out_classifier.npz',data=best_test)
        # else:
        #     np.savez_compressed(os.getcwd()+f'/saved/{cn}_out_classifier.npz',data=out_test)
        # del ClassifierModel
        # gc.collect()
        # torch.cuda.empty_cache()
                
        # # now create ridge
        # dual_data_train_ridge = np.where(np.abs(dual_data_train)<DUAL_THRES,-1,1)
        # totSize = dual_data_train_ridge.shape[1]
        # splitSizes = [totSize//torch.cuda.device_count() for _ in range(torch.cuda.device_count())]
        # splitSizes[-1] = (totSize - sum(splitSizes[:-1])) if len(splitSizes)>1 else splitSizes[-1]
        # RidgeModels = [RidgeNet(2*optObj.n_bus,splitSizes[i]).to(f'cuda:{i}') for i in range(len(splitSizes))]
        # for model in RidgeModels:
        #     for p in model.parameters():
        #         p.data.fill_(0.01)
        # alpha = 1e-3
        # optimRidge = [torch.optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-1) for model in RidgeModels]
        # losses = []
        # FP, FN, TP, TN = [], [], [], []
        # FP_best, FN_best, TP_best, TN_best = 0, dual_data_test.size, 0, 0
        # times = 0
        
        # # carry out training for ridge neural net
        # first_test_done = False
        # best_recorded = False
        # best_epoch = 0
        # start = time()
        # for e in range(epochs):
        #     np.random.seed(e)
        #     random_idx = np.random.choice(inp_data_train.shape[0],BS,replace=False)
        #     rout = np.split(dual_data_train_ridge[random_idx,:],np.cumsum(splitSizes[:-1]),axis=1)
        #     for idx,model in enumerate(RidgeModels):
        #         inp_t = torch.FloatTensor(inp_data_train[random_idx,:2*optObj.n_bus]).to(torch.float32).to(f'cuda:{idx}')
        #         rout_t = torch.FloatTensor(rout[idx]).to(torch.float32).to(f'cuda:{idx}')
        #         loss = F.mse_loss(model(inp_t),rout_t,reduction='mean')
        #         # loss = loss_grad
        #         optimRidge[idx].zero_grad()
        #         loss.backward()
        #         optimRidge[idx].step()
        #         del rout_t
        #         del inp_t
        #         # gc.collect()
        #         # torch.cuda.empty_cache()
        #     losses.append(loss_grad.item())
        #     # test
        #     if (e+1) % 250 == 0:
        #         out_collector = []
        #         for idx,model in enumerate(RidgeModels):
        #             inp = torch.FloatTensor(inp_data_test[:,:2*optObj.n_bus]).to(torch.float32).to(f'cuda:{idx}')
        #             out_collector.append(model(inp).detach().cpu().numpy())
        #             del inp
        #         # gc.collect()
        #         # torch.cuda.empty_cache()
        #         out_test = np.concatenate(out_collector,axis=1)
        #         out_test = np.where(out_test<0,0,1)
        #         tp, tn, fp, fn = out_test[:,nmineq]*dual_data_test[:,nmineq],\
        #             (1-out_test[:,nmineq])*(1-dual_data_test[:,nmineq]),\
        #             out_test[:,nmineq]*(1-dual_data_test[:,nmineq]),\
        #             (1-out_test[:,nmineq])*(dual_data_test[:,nmineq])
        #         tp, tn, fp, fn = tp.sum().item(), tn.sum().item(), fp.sum().item(), fn.sum().item()
        #         FP.append(fp)
        #         TP.append(tp)
        #         FN.append(fn)
        #         TN.append(tn) 
        #         first_test_done = True
        #         # print
        #         if first_test_done:
        #             allv = tp + tn + fp + fn
        #             print(f"For case {cn}, method: ridge, loss on epoch {e+1} is {losses[-1]:.5f}. TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}.",flush=True)
        #             if fn <= FN_best and (tn*100/allv) > 50: # best false negative when true negatives fall below 50 percent
        #                 best_recorded, best_epoch = True, e+1
        #                 TP_best = tp
        #                 TN_best = tn
        #                 FP_best = fp
        #                 FN_best = fn
        #                 best_test = out_test
        #                 for idx,model in enumerate(RidgeModels):
        #                     torch.save(model.state_dict(),os.getcwd()+f'/saved/{cn}_RidgeModel_{idx}.pth')
        #         else:
        #             print(f"For case {cn}, method: ridge, loss on epoch {e+1} is {losses[-1]:.5f}.",flush=True)
        #         # log
        #         if (e+1) == epochs:
        #             if best_recorded:
        #                 times = time() - start
        #                 logfile.write(f"For case {cn}, method: ridge, best performance recorded on epoch {best_epoch}/{epochs}.\nTP: {TP_best*100/allv:.5f}, TN: {TN_best*100/allv:.5f}, FP: {FP_best*100/allv:.5f}, FN: {FN_best*100/allv:.5f}, train_time: {times}s.\n")
        #             else:
        #                 logfile.write(f"For case {cn}, method: ridge. TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}.\n")
        # if not best_recorded:
        #     for idx,model in enumerate(RidgeModels):
        #         torch.save(model.state_dict(),os.getcwd()+f'/saved/{cn}_RidgeModel_{idx}.pth')
        # if best_recorded:
        #     np.savez_compressed(os.getcwd()+f'/saved/{cn}_out_ridge.npz',data=best_test)
        # else:
        #     np.savez_compressed(os.getcwd()+f'/saved/{cn}_out_ridge.npz',data=out_test)
        # del RidgeModels
        # gc.collect()
        # torch.cuda.empty_cache()    
        
        # # now create xgboost
        # if cn == "pglib_opf_case10000_goc":
        #     continue
        # inp_data_train_XGB = inp_data_train.copy()[:,:2*optObj.n_bus]
        # dual_data_train_XGB = np.where(np.abs(dual_data_train)<DUAL_THRES,0,1)
        # dtrain = xgb.DMatrix(inp_data_train_XGB, label=dual_data_train_XGB)
        # num_rounds = 5
        # params = {
        #     'objective': 'binary:logistic',
        #     'n_estimators': num_rounds,
        #     'device': 'cuda:0',  
        #     'learning_rate': 0.1,
        #     'max_depth': 3,
        #     'min_child_weight': 1,
        #     'sampling_method':'gradient_based',
        #     'subsample': 0.8,
        #     'colsample_bytree': 0.8,
        # }
        # model = XGBClassifier(**params)
        # start = time()
        # model.fit(inp_data_train_XGB,dual_data_train_XGB)
        # times = time() - start
        # out_test = model.predict(inp_data_test[:,:2*optObj.n_bus])
        # tp, tn, fp, fn = out_test[:,nmineq]*dual_data_test[:,nmineq],\
        #             (1-out_test[:,nmineq])*(1-dual_data_test[:,nmineq]),\
        #             out_test[:,nmineq]*(1-dual_data_test[:,nmineq]),\
        #             (1-out_test[:,nmineq])*(dual_data_test[:,nmineq])
        # tp, tn, fp, fn = tp.sum().item(), tn.sum().item(), fp.sum().item(), fn.sum().item()
        # allv = tp + tn + fp + fn
        # print(f"For case {cn}, method: xgboost, TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}.",flush=True)
        # # log
        # logfile.write(f"For case {cn}, method: xgboost.\nTP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}, train_time: {times}s.\n\n")
        
        # np.savez_compressed(os.getcwd()+f'/saved/{cn}_out_xgboost.npz',data=out_test)

        # gc.collect()
        
    logfile.close()