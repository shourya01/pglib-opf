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
from ConvexModel import ConvexNet
from ClassifierModel import ClassifierNet
from RidgeModel import RidgeNet
import xgboost as xgb

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import RidgeClassifier

# octave = Oct2Py()
dir_name = 'data/'
MAX_BUS = 10000
DUAL_THRES = 1e-4


if __name__ == "__main__":
    
    # get all cases in current directory
    current_directory = os.getcwd()+'/'
    # current_directory = '/home/sbose/pglib-opf/' # for running on BEBOP
    all_files_and_directories = os.listdir(current_directory)
    # three specific cases
    # todo_cases =  ['pglib_opf_case118_ieee','pglib_opf_case2312_goc',"pglib_opf_case4601_goc","pglib_opf_case10000_goc"]
    todo_cases =  ["pglib_opf_case10000_goc"]
    
    with open(os.getcwd()+'/allcases.pkl','rb') as file:
        data = pickle.load(file)
        
    cases = data['cases']
    casenames = data['casenames']
    cases_new, casenames_new = [], []
    for cs,cn in zip(cases,casenames):
        if cn in todo_cases:
            cases_new.append(cs)
            casenames_new.append(cn)
    cases, casenames = cases_new, casenames_new
    
    # # get all cases in current directory
    # current_directory = os.getcwd()+'/'
    # # current_directory = '/home/sbose/pglib-opf/' # for running on BEBOP
    # all_files_and_directories = os.listdir(current_directory)
    # # three specific cases
    # case_files = [current_directory+i for i in ['pglib_opf_case118_ieee.m','pglib_opf_case2312_goc.m',"pglib_opf_case4601_goc.m","pglib_opf_case10000_goc.m"]]
    # # case_files = [current_directory+i for i in ['pglib_opf_case2312_goc.m',"pglib_opf_case4601_goc.m"]]

    # cases, casenames = [], []
    # cases_full, casenames_full = [], []
    
    # for cf in case_files:
    #     octave.source(current_directory+os.path.basename(cf))
    #     cname = os.path.basename(cf).split('.')[0]
    #     num_buses = None
    #     # determine number of buses in the case from its name
    #     for ci in cname.split('_'):
    #         if 'case' in ci:
    #             num_buses = int(''.join([chr for chr in ci.replace('case','',1) if chr.isdigit()]))
    #     # fitler out cases with more buses than MAX_BUS
    #     if num_buses <= MAX_BUS:
    #         # convert to internal indexing
    #         case_correct_idx = ext2int(loadcase(octave.feval(cname)))
    #         # append
    #         cases.append(case_correct_idx)
    #         casenames.append(cname)
    
    # write
    logfile = open('perf.txt','a')
            
    for cn,this_case in zip(casenames,cases):
        
        # torch.set_default_dtype(torch.bfloat16)
    
        inp_data = np.load(os.getcwd()+'/'+dir_name+f'{cn}_inp.npz')['data']
        dual_data = np.load(os.getcwd()+'/'+dir_name+f'{cn}_dual.npz')['data']
        cost_data = np.load(os.getcwd()+'/'+dir_name+f'{cn}_costs.npz')['data'][:,None]
        
        # modify the input
        optObj = opfSocp(this_case,cn) # generate object
        
        # weights
        WEIGHT_FOR_BINDING_CONSTR = int(100*optObj.n_bus)
        
        # inp_list = []
        # for id in range(inp_data.shape[0]):
        #     # temporary - insert extra flow limit into existing input data (since the original didn't have that)
        #     inp = np.insert(inp_data[id,:],optObj.bus_pd.size+optObj.bus_qd.size+optObj.flow_lim.size,optObj.flow_lim)
        #     inp_list.append(inp)
        # inp_data = np.array(inp_list)
        
        # # load data 
        nn_shape = (inp_data.shape[1],300,150,150,1)
        # ConvexModel = make_data_parallel(ConvexNet(nn_shape,activation=nn.LeakyReLU).to('cuda'),optObj.n_bus)
        # for p in ConvexModel.parameters():
        #     p.data.fill_(0.01)
        # BS = 32 # batch size
        # optimConvex = torch.optim.Adam(ConvexModel.parameters(),lr=1e-4,weight_decay=1e-2)
        epochs = 2000
        losses = []
        FP, FN, TP, TN = [], [], [], []
        
        # partition the data into train and test
        inp_data_train = inp_data[:int(0.8*inp_data.shape[0]),:]
        
        # acquire and preprocess train data
        inp_min, inp_max = inp_data_train.min(), inp_data_train.max() # preprocess - acquire min and max
        inp_data_train = (inp_data_train - inp_min) / (inp_max - inp_min) # preprocess - min-max scaling
        cost_data_train = cost_data[:int(0.8*inp_data.shape[0]),:]
        dual_data_train = dual_data[:int(0.8*inp_data.shape[0]),:]
        dual_data_train = dual_data_train.copy() # save copy for later
        
        # acquire and preprocess test data
        inp_data_test = inp_data[int(0.8*inp_data.shape[0]):,:]
        inp_data_test = (inp_data_test - inp_min) / (inp_max - inp_min) # preprocess - min-max scaling
        cost_data_test = cost_data[int(0.8*inp_data.shape[0]):,:]
        dual_data_test = dual_data[int(0.8*inp_data.shape[0]):,:]
        dual_data_test = np.where(np.abs(dual_data_test)<DUAL_THRES,0.,1.) # preprocess to easily calculate confusion matrix
        
        # # save test inputs
        # np.savez_compressed(os.getcwd()+f'/saved/{cn}_test_inp.npz',data=inp_data_test)
        
        # # vector for nonmodel inequalities
        # nmineq = np.ones(2*optObj.n_bus+4*optObj.n_branch)
        # nmineq[:2*optObj.n_bus] = 0
        # nmineq = nmineq.astype(bool)
        
        # # carry out training for convex neural net
        # first_test_done = False
        # for e in range(epochs):
        #     np.random.seed(e)
        #     random_idx = np.random.choice(inp_data_train.shape[0],BS,replace=False)
        #     inp_t = torch.FloatTensor(inp_data_train[random_idx,:]).to('cuda')
        #     # cost_t = torch.FloatTensor(cost_data_train[random_idx,:]).to('cuda') 
        #     grad_t = torch.FloatTensor(dual_data_train[random_idx,:]).to('cuda')
        #     # loss_cost = F.mse_loss(ConvexModel(inp_t),cost_t,reduction='mean')
        #     wt = torch.FloatTensor(np.where(np.abs(dual_data_train[random_idx,:])<DUAL_THRES,1,WEIGHT_FOR_BINDING_CONSTR)).to('cuda')
        #     loss_grad = F.binary_cross_entropy_with_logits(ConvexModel(inp_t),torch.sigmoid(grad_t),weight=wt)
        #     # loss = loss_grad
        #     optimConvex.zero_grad()
        #     loss_grad.backward()
        #     optimConvex.step()
        #     del wt
        #     del grad_t
        #     losses.append(loss_grad.item())
        #     # test
        #     if (e+1) % 250 == 0:
        #         inp = torch.FloatTensor(inp_data_test).to('cuda')
        #         out_test = ConvexModel(inp).detach().cpu().numpy()
        #         del inp
        #         # out_test = np.where(out_test<0.5*(preprocess_low+preprocess_high),0,1)
        #         out_test = np.where(np.abs(out_test)<DUAL_THRES,0,1)
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
        #             print(f"For case {cn}, method: convex, loss on epoch {e+1} is {losses[-1]:.5f}. TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}.")
        #         else:
        #             print(f"For case {cn}, method: convex, loss on epoch {e+1} is {losses[-1]:.5f}.")
        #         # log
        #         if (e+1) == epochs:
        #             logfile.write(f"For case {cn}, method: convex. TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}.\n")
        # torch.save(ConvexModel.state_dict(),os.getcwd()+f'/saved/{cn}_ConvexModel.pth')
        # np.savez_compressed(os.getcwd()+f'/saved/{cn}_out_convex.npz',data=out_test)
        # del ConvexModel
        # gc.collect()
        # torch.cuda.empty_cache()
                
        # # now create classifier
        # dual_data_train_classifier = np.where(np.abs(dual_data_train)<DUAL_THRES,0,1)
        # ClassifierModel = make_data_parallel(ClassifierNet(nn_shape,activation=nn.LeakyReLU).to('cuda'),optObj.n_bus)
        # for p in ClassifierModel.parameters():
        #     p.data.fill_(0.01)
        # optimClassifier = torch.optim.Adam(ClassifierModel.parameters(),lr=1e-4,weight_decay=1e-2)
        # losses = []
        # FP, FN, TP, TN = [], [], [], []
        
        # # carry out training for classifier neural net
        # first_test_done = False
        # for e in range(epochs):
        #     np.random.seed(e)
        #     random_idx = np.random.choice(inp_data_train.shape[0],BS,replace=False)
        #     inp_t = torch.FloatTensor(inp_data_train[random_idx,:]).to('cuda')
        #     # cost_t = torch.FloatTensor(cost_data_train[random_idx,:]).to('cuda')
        #     probs_t = torch.FloatTensor(dual_data_train_classifier[random_idx,:]).to('cuda')
        #     # loss_cost = F.mse_loss(ConvexModel(inp_t),cost_t,reduction='mean')
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
        #         inp = torch.FloatTensor(inp_data_test).to('cuda')
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
        #             print(f"For case {cn}, method: classifier, loss on epoch {e+1} is {losses[-1]:.5f}. TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}.")
        #         else:
        #             print(f"For case {cn}, method: classifier, loss on epoch {e+1} is {losses[-1]:.5f}.")
        #         # log
        #         if (e+1) == epochs:
        #             logfile.write(f"For case {cn}, method: classifier. TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}.\n")
        # torch.save(ClassifierModel.state_dict(),os.getcwd()+f'/saved/{cn}_ClassifierModel.pth')
        # np.savez_compressed(os.getcwd()+f'/saved/{cn}_out_classifier.npz',data=out_test)
        # del ClassifierModel
        # gc.collect()
        # torch.cuda.empty_cache()
                
        # # now create ridge
        # dual_data_train_ridge = np.where(np.abs(dual_data_train)<DUAL_THRES,-1,1)
        # totSize = dual_data_train_ridge.shape[1]
        # splitSizes = [totSize//torch.cuda.device_count() for _ in range(torch.cuda.device_count())]
        # splitSizes[-1] = (totSize - sum(splitSizes[:-1])) if len(splitSizes)>1 else splitSizes[-1]
        # RidgeModels = [RidgeNet(nn_shape[0],splitSizes[i]).to(f'cuda:{i}') for i in range(len(splitSizes))]
        # for model in RidgeModels:
        #     for p in model.parameters():
        #         p.data.fill_(0.01)
        # alpha = 1e-3
        # optimRidge = [torch.optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-1) for model in RidgeModels]
        # losses = []
        # FP, FN, TP, TN = [], [], [], []
        
        # # carry out training for ridge neural net
        # first_test_done = False
        # for e in range(epochs):
        #     np.random.seed(e)
        #     random_idx = np.random.choice(inp_data_train.shape[0],BS,replace=False)
        #     rout = np.split(dual_data_train_ridge[random_idx,:],np.cumsum(splitSizes[:-1]),axis=1)
        #     for idx,model in enumerate(RidgeModels):
        #         inp_t = torch.FloatTensor(inp_data_train[random_idx,:]).to(f'cuda:{idx}')
        #         # cost_t = torch.FloatTensor(cost_data_train[random_idx,:]).to('cuda')
        #         rout_t = torch.FloatTensor(rout[idx]).to(f'cuda:{idx}')
        #         # loss_cost = F.mse_loss(ConvexModel(inp_t),cost_t,reduction='mean')
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
        #             inp = torch.FloatTensor(inp_data_test).to(f'cuda:{idx}')
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
        #             print(f"For case {cn}, method: ridge, loss on epoch {e+1} is {losses[-1]:.5f}. TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}.")
        #         else:
        #             print(f"For case {cn}, method: ridge, loss on epoch {e+1} is {losses[-1]:.5f}.")
        #         # log
        #         if (e+1) == epochs:
        #             logfile.write(f"For case {cn}, method: ridge. TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}.\n")
        # for idx,model in enumerate(RidgeModels):
        #     torch.save(model.state_dict(),os.getcwd()+f'/saved/{cn}_RidgeModel_{idx}.pth')
        # np.savez_compressed(os.getcwd()+f'/saved/{cn}_out_ridge.npz',data=out_test)
        # del RidgeModels
        # gc.collect()
        # torch.cuda.empty_cache()    
        
        # now create xgboost
        inp_data_train_XGB = inp_data_train.copy()
        dual_data_train_XGB = np.where(np.abs(dual_data_train)<DUAL_THRES,0,1)
        dtrain = xgb.DMatrix(inp_data_train_XGB, label=dual_data_train_XGB)
        # # if all values in column are equal, flip the last element
        # diff = np.diff(dual_data_train_CB, axis=0)
        # constant_columns = np.all(diff == 0, axis=0)
        # variable_column_indices = np.where(~constant_columns)[0]
        # for colidx in variable_column_indices:
        #     dual_data_train_CB[-1,colidx] = 1 - dual_data_train_CB[-2,colidx]
        # define model
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',  # You can also use 'auc', 'error' (for classification error), etc.
            'device': 'cuda:0',  # This parameter specifies the use of the GPU
            'learning_rate': 0.1,
            'max_depth': 4,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
        num_rounds = 5
        bst = xgb.train(params, dtrain, num_rounds)
        dpredict = xgb.DMatrix(inp_data_test)  # Features for prediction
        out_test = bst.predict(dpredict)
        tp, tn, fp, fn = out_test[:,nmineq]*dual_data_test[:,nmineq],\
                    (1-out_test[:,nmineq])*(1-dual_data_test[:,nmineq]),\
                    out_test[:,nmineq]*(1-dual_data_test[:,nmineq]),\
                    (1-out_test[:,nmineq])*(dual_data_test[:,nmineq])
        tp, tn, fp, fn = tp.sum().item(), tn.sum().item(), fp.sum().item(), fn.sum().item()
        allv = tp + tn + fp + fn
        print(f"For case {cn}, method: xgboost, TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}.")
        # log
        logfile.write(f"For case {cn}, method: xgboost. TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}.")
        
        np.savez_compressed(os.getcwd()+f'/saved/{cn}_out_xgboost.npz',data=out_test)
        
        del dpredict
        del dtrain
        gc.collect()
        
    logfile.close()
                
        
        # sklearn ridge
        # rc = RidgeClassifier()
        # modelRidge = MultiOutputClassifier(estimator=rc)
        
        # modelRidge.fit(inp_data_train,dual_data_train_classifier)
        # # predict
        # out_test = modelRidge.predict(inp_data_test)
        # tp, tn, fp, fn = out_test*dual_data_test, (1-out_test)*(1-dual_data_test), out_test*(1-dual_data_test), (1-out_test)*(dual_data_test)
        # tp, tn, fp, fn = tp.sum().item(), tn.sum().item(), fp.sum().item(), fn.sum().item()
        # allv = tp + tn + fp + fn
        # print(f"For case {cn}, method: ridge, loss is {losses[-1]:.5f}. TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}.")
        
        
        # # sklearn decisiontree
        # dtc = DecisionTreeClassifier()
        # modelDTree = MultiOutputClassifier(estimator=dtc)
        
        # modelDTree.fit(inp_data_train,dual_data_train_classifier)
        # # predict
        # out_test = modelDTree.predict(inp_data_test)
        # tp, tn, fp, fn = out_test*dual_data_test, (1-out_test)*(1-dual_data_test), out_test*(1-dual_data_test), (1-out_test)*(dual_data_test)
        # tp, tn, fp, fn = tp.sum().item(), tn.sum().item(), fp.sum().item(), fn.sum().item()
        # allv = tp + tn + fp + fn
        # print(f"For case {cn}, method: decision tree, loss is {losses[-1]:.5f}. TP: {tp*100/allv:.5f}, TN: {tn*100/allv:.5f}, FP: {fp*100/allv:.5f}, FN: {fn*100/allv:.5f}.")