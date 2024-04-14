import numpy as np
from pypower.idx_brch import *
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_cost import *
from pypower.ext2int import ext2int
import math
from typing import List, Dict
from cyipopt import CyIpoptEvaluationError as CEE
import cProfile
from scipy.sparse import csr_matrix, lil_matrix
from copy import deepcopy

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

LARGE_NUMBER = 1e+15 # proxy for infinity
UNIQUE_BUS_FILTER = False # todo

class opfSocp():

    def __init__(self, ppc: Dict, casename: str = 'ppc'):
        
        self.casename = casename
        
        ppc = ext2int(ppc) # convert indices to pypower internal numbering
        self.initSysParams(ppc)
        self.preprocessSysParams()
        self.nodeAndEdgeAttr()
        self.generateIndices()
        self.saveJacobianStructure()
        self.jacobianJIT()
        self.hessianJIT()
        
        self.LARGE_NUMBER = LARGE_NUMBER # save large number
        
    def initSysParams(self, ppc: Dict):
        
        '''
        initSysParams : populates system parameters from pypower case
        '''
        # baseMVA
        self.baseMVA = ppc['baseMVA']
        # bus data
        self.bus_list = ppc['bus'][:,BUS_I].astype(int)
        self.bus_vmin, self.bus_vmax = ppc['bus'][:,VMIN], ppc['bus'][:,VMAX]
        self.bus_pd, self.bus_qd = ppc['bus'][:,PD], ppc['bus'][:,QD]
        self._bus_pd, self._bus_qd = self.bus_pd, self.bus_qd # for mainitaining record of originals
        self.bus_gs, self.bus_bs = ppc['bus'][:,GS], ppc['bus'][:,BS]
        # branches
        self.from_bus, self.to_bus = ppc['branch'][:,F_BUS], ppc['branch'][:,T_BUS]
        self.br_r, self.br_x = ppc['branch'][:,BR_R], ppc['branch'][:,BR_X]
        self.br_b = ppc['branch'][:,BR_B]
        self.tap, self.shift = ppc['branch'][:,TAP], ppc['branch'][:,SHIFT]
        self.angmin, self.angmax = ppc['branch'][:,ANGMIN], ppc['branch'][:,ANGMAX]
        self.flow_lim = ppc['branch'][:,RATE_A]
        self.branch_list = [((f,t),int(l)) for f,t,l in zip(self.from_bus.astype(int),self.to_bus.astype(int),range(self.from_bus.size))]
        # calculate flow limits
        self.flow_lims = ppc['branch'][:,RATE_A]
        # generators
        self.gen_to_bus = ppc['gen'][:,GEN_BUS]
        self.gen_pmax, self.gen_pmin = ppc['gen'][:,PMAX], ppc['gen'][:,PMIN]
        self.gen_qmax, self.gen_qmin = ppc['gen'][:,QMAX], ppc['gen'][:,QMIN]
        self.gen_cost = ppc['gencost'][:,COST:]
        # numbers
        self.n_bus, self.n_gen = self.bus_list.size, self.gen_to_bus.size
        self.n_branch = self.from_bus.size
        
    def preprocessSysParams(self):
        
        # normalization to pu
        self.bus_pd, self.bus_qd, self.bus_gs, self.bus_bs, self.gen_pmax, self.gen_qmax, self.gen_pmin, self.gen_qmin, self.flow_lim = \
        self.bus_pd/self.baseMVA, self.bus_qd/self.baseMVA, self.bus_gs/self.baseMVA, self.bus_bs/self.baseMVA, \
        self.gen_pmax/self.baseMVA, self.gen_qmax/self.baseMVA, self.gen_pmin/self.baseMVA, self.gen_qmin/self.baseMVA, self.flow_lim/self.baseMVA
        self._bus_pd, self._bus_qd = self._bus_pd / self.baseMVA, self._bus_qd / self.baseMVA
        # insert 1 in tap ratios
        self.tap = np.where(np.abs(self.tap)<1e-5,1.,self.tap)
        # convert angle limits to radians
        self.angmin, self.angmax = np.radians(self.angmin), np.radians(self.angmax)
        # convert 0 flow lims to large number
        self.flow_lim = np.where(np.abs(self.flow_lim)<1e-5,1e+3,self.flow_lim)
        
    def generateIndices(self):
        
        # input size
        self.in_size = self.n_bus + 6*self.n_branch + 2*self.n_gen
        raw_idx = np.arange(self.in_size)
        raw_vidx = np.split(raw_idx,np.cumsum([self.n_bus,self.n_branch,self.n_branch,self.n_branch,self.n_branch,self.n_branch,self.n_branch,self.n_gen,self.n_gen])[:-1])
        self.vidx = {
            'U':raw_vidx[0].astype(int),
            'rW':raw_vidx[1].astype(int),
            'iW':raw_vidx[2].astype(int),
            'rSf':raw_vidx[3].astype(int),
            'iSf':raw_vidx[4].astype(int),
            'rSt':raw_vidx[5].astype(int),
            'iSt':raw_vidx[6].astype(int),
            'rSg':raw_vidx[7].astype(int),
            'iSg':raw_vidx[8].astype(int)
        }
        
        # helper functions
        self.out_bus = lambda i: [(w[0][1],w[1]) for w in self.branch_list if w[0][0]==i]
        self.in_bus = lambda i: [(w[0][0],w[1]) for w in self.branch_list if w[0][1]==i]
        self.gen_on_bus = lambda i: [w for w in range(self.n_gen) if self.gen_to_bus[w]==i]
        self.get_branch = lambda f,t: [w[1] for w in self.branch_list if (w[0][0] == f and w[0][1]==t)][0]
        
        # convert helper functions to cached form
        self.get_branch = {w[0]:self.get_branch(*w[0]) for w in self.branch_list}
        self.out_bus = {b:self.out_bus(b) for b in self.bus_list}
        self.in_bus = {b:self.in_bus(b) for b in self.bus_list}
        self.gen_on_bus = {b:self.gen_on_bus(b) for b in self.bus_list}
        
        # constraints
        self.cons_size = 2*self.n_bus + 9*self.n_branch
        raw_coidx = np.arange(self.cons_size)
        raw_cidx = np.split(raw_coidx,np.cumsum([self.n_bus,self.n_bus,self.n_branch,self.n_branch,self.n_branch,self.n_branch,self.n_branch,self.n_branch,self.n_branch,self.n_branch,self.n_branch])[:-1])
        self.cidx = {
            'balance_real':raw_cidx[0].astype(int),
            'balance_reac':raw_cidx[1].astype(int),
            'rSf':raw_cidx[2].astype(int),
            'iSf':raw_cidx[3].astype(int),
            'rSt':raw_cidx[4].astype(int),
            'iSt':raw_cidx[5].astype(int),
            'flow_f':raw_cidx[6].astype(int),
            'flow_t':raw_cidx[7].astype(int),
            'angmin':raw_cidx[8].astype(int),
            'angmax':raw_cidx[9].astype(int),
            'socr':raw_cidx[10].astype(int)
        }
        
    def nodeAndEdgeAttr(self):
        
        # bus arrtributes
        self.busattr, self.brattr = [], []
        for busid in self.bus_list:
            self.busattr.append({'pd':self.bus_pd[busid],'qd':self.bus_qd[busid],'vmin':self.bus_vmin[busid],'vmax':self.bus_vmax[busid],'gs':self.bus_gs[busid],'bs':self.bus_bs[busid]})
        # branch attributes
        for br in self.branch_list:
            br_f,br_t = br[0][0], br[0][1]
            brid = br[1]
            yl, b = 1 / (self.br_r[brid] + 1j * self.br_x[brid]), self.br_b[brid]
            tl = self.tap[brid] * (math.cos(self.shift[brid]) + 1j * math.sin(self.shift[brid]))
            yff, ytt, yft, ytf = (yl + 1j * (b/2)) / (tl*tl.conjugate()), yl + 1j * (b/2), -yl / tl.conjugate(), -yl / tl
            self.brattr.append({'idx':brid,'f':br_f,'t':br_t,'yff':yff,'yft':yft,'ytf':ytf,'ytt':ytt,'vminf':self.bus_vmin[br_f],
                                'vmint':self.bus_vmin[br_t],'vmaxf':self.bus_vmax[br_f],'vmaxt':self.bus_vmax[br_t],'fmax':self.flow_lim[brid],
                                'angmin':self.angmin[brid],'angmax':self.angmax[brid]})
            
    def saveJacobianStructure(self):
        
        jacidx = lil_matrix((self.cons_size,self.in_size))
        self.is_model = []
        self.is_equality = []
        self.is_nonmodel_equality = []
        
        # constraint counter
        cons_counter = 0
        
        # real balance
        for busid in self.bus_list:
            attr = self.busattr[busid]
            for genid in self.gen_on_bus[busid]:
                jacidx[cons_counter,self.vidx['rSg'][genid]] = 1
            jacidx[cons_counter,self.vidx['U'][busid]] = 1
            # add contribution from 'from' buses
            for _,ol in self.out_bus[busid]:
                br_a = self.brattr[ol]
                jacidx[cons_counter,self.vidx['rSf'][br_a['idx']]] = 1
            # add contribution from 'to' buses
            for _,il in self.in_bus[busid]:
                br_a = self.brattr[il]
                jacidx[cons_counter,self.vidx['rSt'][br_a['idx']]] = 1
            cons_counter += 1
            self.is_model.append(0)
            self.is_nonmodel_equality.append(1)
            self.is_equality.append(1)
        
        # reactive balance
        for busid in self.bus_list:
            attr = self.busattr[busid]
            for genid in self.gen_on_bus[busid]:
                jacidx[cons_counter,self.vidx['iSg'][genid]] = 1
            jacidx[cons_counter,self.vidx['U'][busid]] = 1
            # add contribution from 'from' buses
            for _,ol in self.out_bus[busid]:
                br_a = self.brattr[ol]
                jacidx[cons_counter,self.vidx['iSf'][br_a['idx']]] = 1
            # add contribution from 'to' buses
            for _,il in self.in_bus[busid]:
                br_a = self.brattr[il]
                jacidx[cons_counter,self.vidx['iSt'][br_a['idx']]] = 1
            cons_counter += 1
            self.is_model.append(0)
            self.is_nonmodel_equality.append(1)
            self.is_equality.append(1)
            
        # real from flow
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            jacidx[cons_counter,self.vidx['rSf'][attr['idx']]] = 1
            jacidx[cons_counter,self.vidx['U'][attr['f']]] = 1
            jacidx[cons_counter,self.vidx['rW'][attr['idx']]] = 1
            jacidx[cons_counter,self.vidx['iW'][attr['idx']]] = 1
            cons_counter += 1
            self.is_model.append(1)
            self.is_equality.append(0)
            
        # imag from flow
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            jacidx[cons_counter,self.vidx['iSf'][attr['idx']]] = 1
            jacidx[cons_counter,self.vidx['U'][attr['f']]] = 1
            jacidx[cons_counter,self.vidx['rW'][attr['idx']]] = 1
            jacidx[cons_counter,self.vidx['iW'][attr['idx']]] = 1
            cons_counter += 1
            self.is_model.append(1)
            self.is_equality.append(0)
            
        # real to flow
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            jacidx[cons_counter,self.vidx['rSt'][attr['idx']]] = 1
            jacidx[cons_counter,self.vidx['U'][attr['t']]] = 1
            jacidx[cons_counter,self.vidx['rW'][attr['idx']]] = 1
            jacidx[cons_counter,self.vidx['iW'][attr['idx']]] = 1
            cons_counter += 1
            self.is_model.append(1)
            self.is_equality.append(0)
            
        # imag to flow
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            jacidx[cons_counter,self.vidx['iSt'][attr['idx']]] = 1
            jacidx[cons_counter,self.vidx['U'][attr['t']]] = 1
            jacidx[cons_counter,self.vidx['rW'][attr['idx']]] = 1
            jacidx[cons_counter,self.vidx['iW'][attr['idx']]] = 1
            cons_counter += 1
            self.is_model.append(1)
            self.is_equality.append(0)
            
        # from flow limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            jacidx[cons_counter,self.vidx['rSf'][attr['idx']]] = 1
            jacidx[cons_counter,self.vidx['iSf'][attr['idx']]] = 1
            cons_counter += 1 
            self.is_model.append(0)
            self.is_nonmodel_equality.append(0)
            self.is_equality.append(0)
            
        # to flow limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            jacidx[cons_counter,self.vidx['rSt'][attr['idx']]] = 1
            jacidx[cons_counter,self.vidx['iSt'][attr['idx']]] = 1
            cons_counter += 1
            self.is_model.append(0)
            self.is_nonmodel_equality.append(0)
            self.is_equality.append(0)
            
        # minimum angle limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            jacidx[cons_counter,self.vidx['iW'][attr['idx']]] = 1
            jacidx[cons_counter,self.vidx['rW'][attr['idx']]] = 1
            cons_counter += 1
            self.is_model.append(0)
            self.is_nonmodel_equality.append(0)
            self.is_equality.append(0)
            
        # maximum angle limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            jacidx[cons_counter,self.vidx['rW'][attr['idx']]] = 1
            jacidx[cons_counter,self.vidx['iW'][attr['idx']]] = 1
            cons_counter += 1
            self.is_model.append(0)
            self.is_nonmodel_equality.append(0)
            self.is_equality.append(0)
            
        # SOCR limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            jacidx[cons_counter,self.vidx['rW'][attr['idx']]] = 1
            jacidx[cons_counter,self.vidx['iW'][attr['idx']]] = 1
            jacidx[cons_counter,self.vidx['U'][attr['f']]] = 1
            jacidx[cons_counter,self.vidx['U'][attr['t']]] = 1
            cons_counter += 1
            self.is_model.append(1)
            self.is_equality.append(0)
            
        self.jacidx = np.nonzero(jacidx)
        # self.num_indices_per_constr = np.array(self.num_indices_per_constr)
        self.is_model = np.array(self.is_model)
        self.is_nonmodel_equality = np.array(self.is_nonmodel_equality)
        self.is_equality = np.array(self.is_equality)
        
    def jacobianJIT(self):
        
        # precompile jacobian to give JIT-like speed
        
        self.jacflat = {(a,b):i for a,b,i in zip(*self.jacidx,range(self.jacidx[0].size))}
        self.jacsearch = lambda itm: self.jacflat[itm]
        self.jacnum = len(self.jacflat)
        self.jac_coeff = [{'const':0,'linidx':None,'lincoeff':0,'idx':idx} for idx in range(self.jacnum)]
        
        # constraint counter
        cons_counter = 0
        
        # real balance
        for busid in self.bus_list:
            attr = self.busattr[busid]
            for genid in self.gen_on_bus[busid]:
                idx = self.jacsearch((cons_counter,self.vidx['rSg'][genid]))
                self.jac_coeff[idx]['const'] += 1
            idx = self.jacsearch((cons_counter,self.vidx['U'][busid]))
            self.jac_coeff[idx]['const'] += -attr['gs']
            # add contribution from 'from' buses
            for _,ol in self.out_bus[busid]:
                br_a = self.brattr[ol]
                idx = self.jacsearch((cons_counter,self.vidx['rSf'][br_a['idx']]))
                self.jac_coeff[idx]['const'] += -1
            # add contribution from 'to' buses
            for _,il in self.in_bus[busid]:
                br_a = self.brattr[il]
                idx = self.jacsearch((cons_counter,self.vidx['rSt'][br_a['idx']]))
                self.jac_coeff[idx]['const'] += -1
            cons_counter += 1
        
        # reactive balance
        for busid in self.bus_list:
            attr = self.busattr[busid]
            for genid in self.gen_on_bus[busid]:
                idx = self.jacsearch((cons_counter,self.vidx['iSg'][genid]))
                self.jac_coeff[idx]['const'] += 1
            idx = self.jacsearch((cons_counter,self.vidx['U'][busid]))
            self.jac_coeff[idx]['const'] += +attr['bs']
            # add contribution from 'from' buses
            for _,ol in self.out_bus[busid]:
                br_a = self.brattr[ol]
                idx = self.jacsearch((cons_counter,self.vidx['iSf'][br_a['idx']]))
                self.jac_coeff[idx]['const'] += -1
            # add contribution from 'to' buses
            for _,il in self.in_bus[busid]:
                br_a = self.brattr[il]
                idx = self.jacsearch((cons_counter,self.vidx['iSt'][br_a['idx']]))
                self.jac_coeff[idx]['const'] += -1
            cons_counter += 1
            
        # real from flow
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            idx = self.jacsearch((cons_counter,self.vidx['rSf'][attr['idx']]))
            self.jac_coeff[idx]['const'] += 1
            idx = self.jacsearch((cons_counter,self.vidx['U'][attr['f']]))
            self.jac_coeff[idx]['const'] += -attr['yff'].real
            idx = self.jacsearch((cons_counter,self.vidx['rW'][attr['idx']]))
            self.jac_coeff[idx]['const'] += -attr['yft'].real
            idx = self.jacsearch((cons_counter,self.vidx['iW'][attr['idx']]))
            self.jac_coeff[idx]['const'] += -attr['yft'].imag
            cons_counter += 1
            
        # imag from flow
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            idx = self.jacsearch((cons_counter,self.vidx['iSf'][attr['idx']]))
            self.jac_coeff[idx]['const'] += 1
            idx = self.jacsearch((cons_counter,self.vidx['U'][attr['f']]))
            self.jac_coeff[idx]['const'] += attr['yff'].imag
            idx = self.jacsearch((cons_counter,self.vidx['rW'][attr['idx']]))
            self.jac_coeff[idx]['const'] += attr['yft'].imag
            idx = self.jacsearch((cons_counter,self.vidx['iW'][attr['idx']]))
            self.jac_coeff[idx]['const'] += -attr['yft'].real
            cons_counter += 1
            
        # real to flow
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            idx = self.jacsearch((cons_counter,self.vidx['rSt'][attr['idx']]))
            self.jac_coeff[idx]['const'] += 1
            idx = self.jacsearch((cons_counter,self.vidx['U'][attr['t']]))
            self.jac_coeff[idx]['const'] += -attr['ytt'].real
            idx = self.jacsearch((cons_counter,self.vidx['rW'][attr['idx']]))
            self.jac_coeff[idx]['const'] += -attr['ytf'].real
            idx = self.jacsearch((cons_counter,self.vidx['iW'][attr['idx']]))
            self.jac_coeff[idx]['const'] += attr['ytf'].imag
            cons_counter += 1
            
        # imag to flow
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            idx = self.jacsearch((cons_counter,self.vidx['iSt'][attr['idx']]))
            self.jac_coeff[idx]['const'] += 1
            idx = self.jacsearch((cons_counter,self.vidx['U'][attr['t']]))
            self.jac_coeff[idx]['const'] += attr['ytt'].imag
            idx = self.jacsearch((cons_counter,self.vidx['rW'][attr['idx']]))
            self.jac_coeff[idx]['const'] += attr['ytf'].imag
            idx = self.jacsearch((cons_counter,self.vidx['iW'][attr['idx']]))
            self.jac_coeff[idx]['const'] += attr['ytf'].real
            cons_counter += 1
            
        # from flow limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            idx = self.jacsearch((cons_counter,self.vidx['rSf'][attr['idx']]))
            self.jac_coeff[idx]['linidx'] = self.vidx['rSf'][attr['idx']]
            self.jac_coeff[idx]['lincoeff'] = 2
            idx = self.jacsearch((cons_counter,self.vidx['iSf'][attr['idx']]))
            self.jac_coeff[idx]['linidx'] = self.vidx['iSf'][attr['idx']]
            self.jac_coeff[idx]['lincoeff'] = 2
            cons_counter += 1 
            
        # to flow limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            idx = self.jacsearch((cons_counter,self.vidx['rSt'][attr['idx']]))
            self.jac_coeff[idx]['linidx'] = self.vidx['rSt'][attr['idx']]
            self.jac_coeff[idx]['lincoeff'] = 2
            idx = self.jacsearch((cons_counter,self.vidx['iSt'][attr['idx']]))
            self.jac_coeff[idx]['linidx'] = self.vidx['iSt'][attr['idx']]
            self.jac_coeff[idx]['lincoeff'] = 2
            cons_counter += 1
            
        # minimum angle limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            idx = self.jacsearch((cons_counter,self.vidx['iW'][attr['idx']]))
            self.jac_coeff[idx]['const'] += 1
            idx = self.jacsearch((cons_counter,self.vidx['rW'][attr['idx']]))
            self.jac_coeff[idx]['const'] += - np.tan(attr['angmax'])
            cons_counter += 1
            
        # maximum angle limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            idx = self.jacsearch((cons_counter,self.vidx['rW'][attr['idx']]))
            self.jac_coeff[idx]['const'] += np.tan(attr['angmin'])
            idx = self.jacsearch((cons_counter,self.vidx['iW'][attr['idx']]))
            self.jac_coeff[idx]['const'] += - 1
            cons_counter += 1
            
        # SOCR limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            idx = self.jacsearch((cons_counter,self.vidx['rW'][attr['idx']]))
            self.jac_coeff[idx]['linidx'] = self.vidx['rW'][attr['idx']]
            self.jac_coeff[idx]['lincoeff'] = 2
            idx = self.jacsearch((cons_counter,self.vidx['iW'][attr['idx']]))
            self.jac_coeff[idx]['linidx'] = self.vidx['iW'][attr['idx']]
            self.jac_coeff[idx]['lincoeff'] = 2
            idx = self.jacsearch((cons_counter,self.vidx['U'][attr['f']]))
            self.jac_coeff[idx]['linidx'] = self.vidx['U'][attr['t']]
            self.jac_coeff[idx]['lincoeff'] = -1
            idx = self.jacsearch((cons_counter,self.vidx['U'][attr['t']]))
            self.jac_coeff[idx]['linidx'] = self.vidx['U'][attr['f']]
            self.jac_coeff[idx]['lincoeff'] = -1
            cons_counter += 1
        
        # constant part of jacobian
        self.jac_const = np.array([itm['const'] for itm in self.jac_coeff])
        self.jac_emb_idx = np.array([itm['idx'] for itm in self.jac_coeff if itm['linidx'] is not None])
        self.jac_lin_idx = np.array([itm['linidx'] for itm in self.jac_coeff if itm['linidx'] is not None]).astype(int)
        self.jac_lin_coeff = np.array([itm['lincoeff'] for itm in self.jac_coeff if itm['linidx'] is not None])
        
    def hessianJIT(self):
        
        hidx = lil_matrix((self.in_size,self.in_size))
        n_dual = self.cons_size + 1 # constraints and objective function

        idx_ff, idx_tf = self.cidx['flow_f'], self.cidx['flow_t']
        idx_socr = self.cidx['socr']
        
        # objective's constribution to hessian
        for gidx in range(self.n_gen):
            if np.abs(self.gen_cost[gidx,0]) > 1e-5: # quadratic cost
                hidx[self.vidx['rSg'][gidx],self.vidx['rSg'][gidx]] = 1
                
        # from flow limits' contribution to hessian
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            lam_idx = idx_ff[brid]
            
            hidx[self.vidx['rSf'][attr['idx']],self.vidx['rSf'][attr['idx']]] = 1
            hidx[self.vidx['iSf'][attr['idx']],self.vidx['iSf'][attr['idx']]] = 1
            
        # to flow limits' contribution to hessian
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            lam_idx = idx_tf[brid]
            
            hidx[self.vidx['rSt'][attr['idx']],self.vidx['rSt'][attr['idx']]] = 1
            hidx[self.vidx['iSt'][attr['idx']],self.vidx['iSt'][attr['idx']]] = 1
            
        # SOCR limits' contribution to hessian
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            lam_idx = idx_socr[brid]
            
            hidx[self.vidx['rW'][attr['idx']],self.vidx['rW'][attr['idx']]] = 1
            hidx[self.vidx['iW'][attr['idx']],self.vidx['iW'][attr['idx']]] = 1
            r,c = self.vidx['U'][attr['f']],self.vidx['U'][attr['t']]
            # keep hessian lower-triangular
            if r>=c:
                hidx[r,c] = 1
            else:
                hidx[c,r] = 1
        
        # save sparsity index
        self.hesidx = hidx.nonzero()
        
        # now build the linear factor
        self.HesTemplate = lil_matrix((self.hesidx[0].size,n_dual))
        rc_dict = {(r,c):i for r,c,i in zip(*self.hesidx,range(self.hesidx[0].size))}
        rc_to_idx = lambda r,c: rc_dict[(r,c)]
        
        # objective's constribution to hessian
        for gidx in range(self.n_gen):
            if np.abs(self.gen_cost[gidx,0]) > 1e-5: # quadratic cost
                self.HesTemplate[rc_to_idx(self.vidx['rSg'][gidx],self.vidx['rSg'][gidx]),0] += 2*self.gen_cost[gidx,0]*self.baseMVA*self.baseMVA
                
        # from flow limits' contribution to hessian
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            lam_idx = idx_ff[brid]
            
            self.HesTemplate[rc_to_idx(self.vidx['rSf'][attr['idx']],self.vidx['rSf'][attr['idx']]),lam_idx+1] += 2
            self.HesTemplate[rc_to_idx(self.vidx['iSf'][attr['idx']],self.vidx['iSf'][attr['idx']]),lam_idx+1] += 2
            
        # to flow limits' contribution to hessian
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            lam_idx = idx_tf[brid]
            
            self.HesTemplate[rc_to_idx(self.vidx['rSt'][attr['idx']],self.vidx['rSt'][attr['idx']]),lam_idx+1] += 2
            self.HesTemplate[rc_to_idx(self.vidx['iSt'][attr['idx']],self.vidx['iSt'][attr['idx']]),lam_idx+1] += 2
            
        # SOCR limits' contribution to hessian
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            lam_idx = idx_socr[brid]
            
            self.HesTemplate[rc_to_idx(self.vidx['rW'][attr['idx']],self.vidx['rW'][attr['idx']]),lam_idx+1] += 2
            self.HesTemplate[rc_to_idx(self.vidx['iW'][attr['idx']],self.vidx['iW'][attr['idx']]),lam_idx+1] += 2
            r,c = self.vidx['U'][attr['f']],self.vidx['U'][attr['t']]
            if r>=c:
                self.HesTemplate[rc_to_idx(r,c),lam_idx+1] += -1
            else:
                self.HesTemplate[rc_to_idx(c,r),lam_idx+1] += -1
        
        # now convert into matrix multiplication form H*v in csr format
        self.HesTemplate = self.HesTemplate.tocsr()
        
    def jacobianstructure(self):
        
        return self.jacidx
    
    def hessianstructure(self):
        
        return self.hesidx
        
    def objective(self,x):
        
        Pg = x[self.vidx['rSg']]
        obj_fn = 0
        
        for gidx in range(self.n_gen):
            if np.abs(self.gen_cost[gidx,0]) > 1e-5: # quadratic cost
                obj_fn = obj_fn + self.gen_cost[gidx,0]*math.pow(Pg[gidx]*self.baseMVA,2)
            if np.abs(self.gen_cost[gidx,1]) > 1e-5: # linear cost
                obj_fn = obj_fn + self.gen_cost[gidx,1]*Pg[gidx]*self.baseMVA
            if np.abs(self.gen_cost[gidx,2]) > 1e-5: # constant cost
                obj_fn = obj_fn + self.gen_cost[gidx,2]
                
        return obj_fn
    
    def gradient(self, x):
        
        grad = np.zeros_like(x)
        gi = lambda g: self.vidx['rSg'][g]
        
        for gidx in range(self.n_gen):
            if np.abs(self.gen_cost[gidx,0]) > 1e-5: # quadratic cost
                grad[gi(gidx)] += 2*self.gen_cost[gidx,0]*x[gi(gidx)]*self.baseMVA*self.baseMVA
            if np.abs(self.gen_cost[gidx,1]) > 1e-5: # linear cost
                grad[gi(gidx)] += self.gen_cost[gidx,1]*self.baseMVA
                
        return grad
    
    def constraints(self, x):
        
        U, rW, iW, rSf, iSf, rSt, iSt, rSg, iSg = x[self.vidx['U']], x[self.vidx['rW']], x[self.vidx['iW']],\
            x[self.vidx['rSf']], x[self.vidx['iSf']], x[self.vidx['rSt']], x[self.vidx['iSt']], x[self.vidx['rSg']], x[self.vidx['iSg']]
        constr = []
        
        # real balance
        for busid in self.bus_list:
            cons = 0
            attr = self.busattr[busid]
            for genid in self.gen_on_bus[busid]:
                cons += rSg[genid]
            cons -= attr['pd']
            cons -= attr['gs'] * U[busid]
            # add contribution 'from' buses
            pfbus, ptbus = 0, 0
            for _,ol in self.out_bus[busid]:
                br_a = self.brattr[ol]
                pfbus += rSf[br_a['idx']]
            for _,il in self.in_bus[busid]:
                br_a = self.brattr[il]
                ptbus += rSt[br_a['idx']]
            cons -= (pfbus+ptbus)
            constr.append(cons)
            
        # reactive balance
        for busid in self.bus_list:
            cons = 0
            attr = self.busattr[busid]
            for genid in self.gen_on_bus[busid]:
                cons += iSg[genid]
            cons -= attr['qd']
            cons += attr['bs'] * U[busid]
            # add contribution 'from' buses
            qfbus, qtbus = 0, 0
            for _,ol in self.out_bus[busid]:
                br_a = self.brattr[ol]
                qfbus += iSf[br_a['idx']]
            for _,il in self.in_bus[busid]:
                br_a = self.brattr[il]
                qtbus += iSt[br_a['idx']]
            cons -= (qfbus+qtbus)
            constr.append(cons)
            
        # real from flow
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            constr.append(rSf[brid] - attr['yff'].real*U[attr['f']] - attr['yft'].real*rW[attr['idx']] - attr['yft'].imag*iW[attr['idx']])
            
        # imag from flow
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            constr.append(iSf[brid] + attr['yff'].imag*U[attr['f']] - attr['yft'].real*iW[attr['idx']] + attr['yft'].imag*rW[attr['idx']])
            
        # real to flow
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            constr.append(rSt[brid] - attr['ytt'].real*U[attr['t']] - attr['ytf'].real*rW[attr['idx']] + attr['ytf'].imag*iW[attr['idx']])
            
        # imag to flow
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            constr.append(iSt[brid] + attr['ytt'].imag*U[attr['t']] + attr['ytf'].real*iW[attr['idx']] + attr['ytf'].imag*rW[attr['idx']])
            
        # from flow limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            constr.append(math.pow(rSf[brid],2)+math.pow(iSf[brid],2)-math.pow(attr['fmax'],2))
        
        # to flow limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            constr.append(math.pow(rSt[brid],2)+math.pow(iSt[brid],2)-math.pow(attr['fmax'],2))
            
        # minimum angle limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            constr.append(iW[brid] - np.tan(attr['angmax'])*rW[brid])
            
        # maximum angle limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            constr.append(np.tan(attr['angmin'])*rW[brid] - iW[brid])
            
        # SOCR limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            constr.append(math.pow(rW[brid],2) + math.pow(iW[brid],2) - U[attr['f']]*U[attr['t']])

        return np.array(constr)
            
    def jacobian(self, x):
        
        if not np.isfinite(x).sum():
            raise CEE("In x for jacobian calculations")
        
        # JIT
        z = np.zeros_like(self.jac_const)
        z[self.jac_emb_idx] = x[self.jac_lin_idx]*self.jac_lin_coeff
        return self.jac_const + z
    
    def hessian(self, x, _lam, _objfac):
        
        if not np.isfinite(x).sum():
            raise CEE("In x for hessian calculations")
        if not np.isfinite(_lam).sum():
            raise CEE("In lambda for hessian calculations")
        if not np.isfinite(_objfac).sum():
            raise CEE("In obj_lambda for hessian calculations")
        
        concat_lam = np.insert(_lam,0,_objfac)

        # JIT 
        return self.HesTemplate.dot(concat_lam).ravel()
        
    
    def calc_cons_bounds(self):
        
        ub = np.zeros(self.cons_size)
        lb = -self.LARGE_NUMBER*np.ones(self.cons_size) # proxy for -inf
        
        # set zero lower bound for equalities
        lb[self.cidx['balance_real']] = 0
        lb[self.cidx['balance_reac']] = 0
        lb[self.cidx['rSf']] = 0
        lb[self.cidx['iSf']] = 0
        lb[self.cidx['rSt']] = 0
        lb[self.cidx['iSt']] = 0
        
        return ub, lb
        
    def calc_var_bounds(self):
        
        ub = np.ones(self.in_size)
        lb = np.ones(self.in_size)
        
        # voltage squared
        ub[self.vidx['U']] = np.square(self.bus_vmax)
        lb[self.vidx['U']] = np.square(self.bus_vmin)
        
        # flow terms
        ub[self.vidx['rSf']] = self.flow_lim
        lb[self.vidx['rSf']] = -self.flow_lim
        ub[self.vidx['iSf']] = self.flow_lim
        lb[self.vidx['iSf']] = -self.flow_lim
        ub[self.vidx['rSt']] = self.flow_lim
        lb[self.vidx['rSt']] = -self.flow_lim
        ub[self.vidx['iSt']] = self.flow_lim
        lb[self.vidx['iSt']] = -self.flow_lim
        
        # bounds for w-terms 
        upper_w_v = np.array([i*j for i,j in zip([self.brattr[brid]['vmaxf'] for brid in range(self.n_branch)],[self.brattr[brid]['vmaxt'] for brid in range(self.n_branch)])])
        lower_w_v = -upper_w_v
        
        # real W
        ub[self.vidx['rW']] = upper_w_v
        lb[self.vidx['rW']] = lower_w_v
        # imag W
        ub[self.vidx['iW']] = upper_w_v
        lb[self.vidx['iW']] = lower_w_v
        # real generation
        ub[self.vidx['rSg']] = self.gen_pmax
        lb[self.vidx['rSg']] = self.gen_pmin
        # reactive generation'
        ub[self.vidx['iSg']] = self.gen_qmax
        lb[self.vidx['iSg']] = self.gen_qmin
        
        return ub, lb
    
    def calc_x0_flatstart(self):
        
        # flat start
        
        x0 = np.zeros(self.in_size)
        # voltage squared
        x0[self.vidx['U']] = 1
        # real forward flow
        x0[self.vidx['rSf']] = 0
        # imag forward flow
        x0[self.vidx['iSf']] = 0
        # real back flow
        x0[self.vidx['rSt']] = 0
        # imag back flow
        x0[self.vidx['iSt']] = 0
        # real W
        x0[self.vidx['rW']] = 1
        # imag W
        x0[self.vidx['iW']] = 0
        # real gen
        x0[self.vidx['rSg']] = self.gen_pmin
        # react gen
        x0[self.vidx['iSg']] = 0.5*(self.gen_qmin+self.gen_qmax)
        
        return x0
    
    def calc_x0_zero(self):
        
        return np.zeros(self.in_size)
    
    def change_loads(self, pd, qd):
        
        for idx in self.bus_list:
            self.busattr[idx]['pd'] = pd[idx]
            self.busattr[idx]['qd'] = qd[idx]
            
        self.bus_pd, self.bus_qd = pd, qd
            
    def get_loads(self):
        
        return self._bus_pd.copy(), self._bus_qd.copy()
    
    def vars_calculator(self,x):
        
        U, rW, iW, rSg, iSg = x[self.vidx['U']], x[self.vidx['rW']], x[self.vidx['iW']], x[self.vidx['rSg']], x[self.vidx['iSg']]
        
        rSf, iSf, rSt, iSt = x[self.vidx['rSf']], x[self.vidx['iSf']], x[self.vidx['rSt']], x[self.vidx['iSt']]
            
        Sg = rSg + 1j * iSg
        W = rW + 1j * iW
        Sf = rSf + 1j * iSf
        St = rSt + 1j * iSt
        
        return U, W, Sg, Sf, St          