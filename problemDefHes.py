import numpy as np
from pypower.idx_brch import *
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_cost import *
from pypower.ext2int import ext2int
import math
from typing import List, Dict
from cyipopt import CyIpoptEvaluationError as CEE

LARGE_NUMBER = 1e+15 # proxy for infinity

class opfSocp():

    def __init__(self, ppc: Dict, casename: str = 'ppc'):
        
        self.casename = casename
        
        ppc = ext2int(ppc) # convert indices to pypower internal numbering
        self.initSysParams(ppc)
        self.preprocessSysParams()
        self.nodeAndEdgeAttr()
        self.generateIndices()
        
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
        # insert 1 in tap ratios
        self.tap = np.where(np.abs(self.tap)<1e-5,1.,self.tap)
        # convert angle limits to radians
        self.angmin, self.angmax = np.radians(self.angmin), np.radians(self.angmax)
        # convert 0 flow lims to large number
        self.flow_lim = np.where(np.abs(self.flow_lim)<1e-5,1e+3,self.flow_lim)
        
    def generateIndices(self):
        
        # input size
        self.in_size = self.n_bus + 2*self.n_branch + 2*self.n_gen
        raw_idx = np.arange(self.in_size)
        raw_vidx = np.split(raw_idx,np.cumsum([self.n_bus,self.n_branch,self.n_branch,self.n_gen,self.n_gen])[:-1])
        self.vidx = {
            'U':raw_vidx[0].astype(int),
            'rW':raw_vidx[1].astype(int),
            'iW':raw_vidx[2].astype(int),
            'rSg':raw_vidx[3].astype(int),
            'iSg':raw_vidx[4].astype(int)
        }
        
        # helper functions
        self.out_bus = lambda i: [w[0][1] for w in self.branch_list if w[0][0]==i]
        self.in_bus = lambda i: [w[0][0] for w in self.branch_list if w[0][1]==i]
        self.gen_on_bus = lambda i: [w for w in range(self.n_gen) if self.gen_to_bus[w]==i]
        self.get_branch = lambda f,t: [w[1] for w in self.branch_list if (w[0][0] == f and w[0][1]==t)][0]
        
        # constraints
        self.cons_size = 2*self.n_bus + 5*self.n_branch
        raw_coidx = np.arange(self.cons_size)
        raw_cidx = np.split(raw_coidx,np.cumsum([self.n_bus,self.n_bus,self.n_branch,self.n_branch,self.n_branch,self.n_branch,self.n_branch])[:-1])
        self.cidx = {
            'balance_real':raw_cidx[0].astype(int),
            'balance_reac':raw_cidx[1].astype(int),
            'flow_f':raw_cidx[2].astype(int),
            'flow_t':raw_cidx[3].astype(int),
            'angmin':raw_cidx[4].astype(int),
            'angmax':raw_cidx[5].astype(int),
            'socr':raw_cidx[6].astype(int)
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
        
    def objective(self,x):
        
        Pg = x[self.vidx['rSg']]
        obj_fn = 0
        
        for gidx in range(self.n_gen):
            if np.abs(self.gen_cost[gidx,0]) > 1e-5: # quadratic cost
                obj_fn = obj_fn + self.gen_cost[gidx,0]*math.pow(Pg[gidx],2)
            if np.abs(self.gen_cost[gidx,1]) > 1e-5: # linear cost
                obj_fn = obj_fn + self.gen_cost[gidx,1]*Pg[gidx]
            if np.abs(self.gen_cost[gidx,2]) > 1e-5: # constant cost
                obj_fn = obj_fn + self.gen_cost[gidx,2]
                
        return obj_fn
    
    def gradient(self, x):
        
        grad = np.zeros_like(x)
        gi = lambda g: self.vidx['rSg'][g]
        
        for gidx in range(self.n_gen):
            if np.abs(self.gen_cost[gidx,0]) > 1e-5: # quadratic cost
                grad[gi(gidx)] += 2*self.gen_cost[gidx,0]*x[gi(gidx)]
            if np.abs(self.gen_cost[gidx,1]) > 1e-5: # linear cost
                grad[gi(gidx)] += self.gen_cost[gidx,1]
                
        return grad
    
    def constraints(self, x):
        
        U, rW, iW, rSg, iSg = x[self.vidx['U']], x[self.vidx['rW']], x[self.vidx['iW']], x[self.vidx['rSg']], x[self.vidx['iSg']]
        constr = []
        
        # real balance
        for busid in self.bus_list:
            cons = 0
            attr = self.busattr[busid]
            for genid in self.gen_on_bus(busid):
                cons += rSg[genid]
            cons -= attr['pd']
            cons -= attr['gs'] * U[busid]
            # add contribution 'from' buses
            pfbus, ptbus = 0, 0
            for obus in self.out_bus(busid):
                br_a = self.brattr[self.get_branch(busid,obus)]
                pfbus += br_a['yff'].real*U[br_a['f']] + br_a['yft'].real*rW[br_a['idx']] + br_a['yft'].imag*iW[br_a['idx']] # math
            for ibus in self.in_bus(busid):
                br_a = self.brattr[self.get_branch(ibus,busid)]
                ptbus += br_a['ytt'].real*U[br_a['t']] + br_a['ytf'].real*rW[br_a['idx']] - br_a['ytf'].imag*iW[br_a['idx']] # math
            cons -= (pfbus+ptbus)
            constr.append(cons)
            
        # reactive balance
        for busid in self.bus_list:
            cons = 0
            attr = self.busattr[busid]
            for genid in self.gen_on_bus(busid):
                cons += iSg[genid]
            cons -= attr['qd']
            cons += attr['bs'] * U[busid]
            # add contribution 'from' buses
            qfbus, qtbus = 0, 0
            for obus in self.out_bus(busid):
                br_a = self.brattr[self.get_branch(busid,obus)]
                qfbus += -br_a['yff'].imag*U[br_a['f']] + br_a['yft'].real*iW[br_a['idx']] - br_a['yft'].imag*rW[br_a['idx']] # math
            for ibus in self.in_bus(busid):
                br_a = self.brattr[self.get_branch(ibus,busid)]
                qtbus += -br_a['ytt'].imag*U[br_a['t']] - br_a['ytf'].real*iW[br_a['idx']] - br_a['ytf'].imag*rW[br_a['idx']] # math
            cons -= (qfbus+qtbus)
            constr.append(cons)
            
        # from flow limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            rSf = attr['yff'].real*U[attr['f']] + attr['yft'].real*rW[attr['idx']] + attr['yft'].imag*iW[attr['idx']] # math
            iSf = -attr['yff'].imag*U[attr['f']] + attr['yft'].real*iW[attr['idx']] - attr['yft'].imag*rW[attr['idx']] # math
            constr.append(math.pow(rSf,2)+math.pow(iSf,2)-math.pow(attr['fmax'],2))
        
        # to flow limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            rSt = attr['ytt'].real*U[attr['t']] + attr['ytf'].real*rW[attr['idx']] - attr['ytf'].imag*iW[attr['idx']] # math
            iSt = -attr['ytt'].imag*U[attr['t']] - attr['ytf'].real*iW[attr['idx']] - attr['ytf'].imag*rW[attr['idx']] # math
            constr.append(math.pow(rSt,2)+math.pow(iSt,2)-math.pow(attr['fmax'],2))
            
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
        
        jac = np.zeros(shape=(self.cons_size,self.in_size))
        cons_counter = 0
        U, rW, iW = x[self.vidx['U']], x[self.vidx['rW']], x[self.vidx['iW']]
        
        # real balance
        for busid in self.bus_list:
            attr = self.busattr[busid]
            for genid in self.gen_on_bus(busid):
                jac[cons_counter,self.vidx['rSg'][genid]] += 1
            jac[cons_counter,self.vidx['U'][busid]] += -attr['gs']
            # add contribution from 'from' buses
            for obus in self.out_bus(busid):
                br_a = self.brattr[self.get_branch(busid,obus)]
                jac[cons_counter,self.vidx['U'][br_a['f']]] += -br_a['yff'].real
                jac[cons_counter,self.vidx['rW'][br_a['idx']]] += -br_a['yft'].real
                jac[cons_counter,self.vidx['iW'][br_a['idx']]] += -br_a['yft'].imag
            # add contribution from 'to' buses
            for ibus in self.in_bus(busid):
                br_a = self.brattr[self.get_branch(ibus,busid)]
                jac[cons_counter,self.vidx['U'][br_a['t']]] += -br_a['ytt'].real
                jac[cons_counter,self.vidx['rW'][br_a['idx']]] += -br_a['ytf'].real
                jac[cons_counter,self.vidx['iW'][br_a['idx']]] += br_a['ytf'].imag
            cons_counter += 1
        
        # reactive balance
        for busid in self.bus_list:
            attr = self.busattr[busid]
            for genid in self.gen_on_bus(busid):
                jac[cons_counter,self.vidx['iSg'][genid]] += 1
            jac[cons_counter,self.vidx['U'][busid]] += +attr['bs']
            # add contribution from 'from' buses
            for obus in self.out_bus(busid):
                br_a = self.brattr[self.get_branch(busid,obus)]
                jac[cons_counter,self.vidx['U'][br_a['f']]] += br_a['yff'].imag
                jac[cons_counter,self.vidx['rW'][br_a['idx']]] += br_a['yft'].imag
                jac[cons_counter,self.vidx['iW'][br_a['idx']]] += -br_a['yft'].real
            # add contribution from 'to' buses
            for ibus in self.in_bus(busid):
                br_a = self.brattr[self.get_branch(ibus,busid)]
                jac[cons_counter,self.vidx['U'][br_a['t']]] += br_a['ytt'].imag
                jac[cons_counter,self.vidx['rW'][br_a['idx']]] += br_a['ytf'].imag
                jac[cons_counter,self.vidx['iW'][br_a['idx']]] += br_a['ytf'].real
            cons_counter += 1
            
        # from flow limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            rSf = attr['yff'].real*U[attr['f']] + attr['yft'].real*rW[attr['idx']] + attr['yft'].imag*iW[attr['idx']] # math
            iSf = -attr['yff'].imag*U[attr['f']] + attr['yft'].real*iW[attr['idx']] - attr['yft'].imag*rW[attr['idx']] # math
            # add jacobian terms
            jac[cons_counter,self.vidx['U'][attr['f']]] += 2*rSf*attr['yff'].real
            jac[cons_counter,self.vidx['rW'][attr['idx']]] += 2*rSf*attr['yft'].real
            jac[cons_counter,self.vidx['iW'][attr['idx']]] += 2*rSf*attr['yft'].imag
            jac[cons_counter,self.vidx['U'][attr['f']]] += -2*iSf*attr['yff'].imag
            jac[cons_counter,self.vidx['rW'][attr['idx']]] += -2*iSf*attr['yft'].imag
            jac[cons_counter,self.vidx['iW'][attr['idx']]] += 2*iSf*attr['yft'].real
            cons_counter += 1 
            
        # to flow limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            rSt = attr['ytt'].real*U[attr['t']] + attr['ytf'].real*rW[attr['idx']] - attr['ytf'].imag*iW[attr['idx']] # math
            iSt = -attr['ytt'].imag*U[attr['t']] - attr['ytf'].real*iW[attr['idx']] - attr['ytf'].imag*rW[attr['idx']] # math
            # add jacobian terms
            jac[cons_counter,self.vidx['U'][attr['t']]] += 2*rSt*attr['ytt'].real
            jac[cons_counter,self.vidx['rW'][attr['idx']]] += 2*rSt*attr['ytf'].real
            jac[cons_counter,self.vidx['iW'][attr['idx']]] += -2*rSt*attr['ytf'].imag
            jac[cons_counter,self.vidx['U'][attr['t']]] += -2*iSt*attr['yff'].imag
            jac[cons_counter,self.vidx['rW'][attr['idx']]] += -2*iSt*attr['ytf'].imag
            jac[cons_counter,self.vidx['iW'][attr['idx']]] += -2*iSt*attr['ytf'].real
            cons_counter += 1
            
        # minimum angle limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            jac[cons_counter,self.vidx['iW'][brid]] += 1
            jac[cons_counter,self.vidx['rW'][brid]] += - np.tan(attr['angmax'])
            cons_counter += 1
            
        # maximum angle limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            jac[cons_counter,self.vidx['rW'][brid]] += np.tan(attr['angmin'])
            jac[cons_counter,self.vidx['iW'][brid]] += -1
            cons_counter += 1
            
        # SOCR limits
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            jac[cons_counter,self.vidx['rW'][brid]] += 2*rW[brid]
            jac[cons_counter,self.vidx['iW'][brid]] += 2*iW[brid]
            jac[cons_counter,self.vidx['U'][attr['f']]] += -U[attr['t']]
            jac[cons_counter,self.vidx['U'][attr['t']]] += -U[attr['f']]
            cons_counter += 1
            
        return jac
    
    # def hessianstructure(self):
        
    #     return np.nonzero(np.tril(np.ones((self.in_size, self.in_size))))
    
    def hessian(self, x, _lam, _objfac):
        
        if not np.isfinite(x).sum():
            raise CEE("In x for hessian calculations")
        if not np.isfinite(_lam).sum():
            raise CEE("In lambda for hessian calculations")
        if not np.isfinite(_objfac).sum():
            raise CEE("In obj_lambda for hessian calculations")
        
        hes = np.zeros(shape=(self.in_size,self.in_size))
        idx_ff, idx_tf = self.cidx['flow_f'], self.cidx['flow_t']
        idx_socr = self.cidx['socr']
        
        # objective's constribution to hessian
        for gidx in range(self.n_gen):
            if np.abs(self.gen_cost[gidx,0]) > 1e-5: # quadratic cost
                hes[self.vidx['rSg'][gidx],self.vidx['rSg'][gidx]] += 2*_objfac*self.gen_cost[gidx,0]
                
        # from flow limits' contribution to hessian
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            _lam_cur = _lam[idx_ff[brid]]
            
            # rSf = attr['yff'].real*U[attr['f']] + attr['yft'].real*rW[attr['idx']] + attr['yft'].imag*iW[attr['idx']] # math
            # iSf = -attr['yff'].imag*U[attr['f']] + attr['yft'].real*iW[attr['idx']] - attr['yft'].imag*rW[attr['idx']] # math
            
            hes[self.vidx['U'][attr['f']],self.vidx['U'][attr['f']]] += 2*_lam_cur*attr['yff'].real*attr['yff'].real
            hes[self.vidx['U'][attr['f']],self.vidx['rW'][attr['idx']]] += 2*_lam_cur*attr['yff'].real*attr['yft'].real
            hes[self.vidx['U'][attr['f']],self.vidx['iW'][attr['idx']]] += 2*_lam_cur*attr['yff'].real*attr['yft'].imag
            
            hes[self.vidx['rW'][attr['idx']],self.vidx['U'][attr['f']]] += 2*_lam_cur*attr['yft'].real*attr['yff'].real
            hes[self.vidx['rW'][attr['idx']],self.vidx['rW'][attr['idx']]] += 2*_lam_cur*attr['yft'].real*attr['yft'].real
            hes[self.vidx['rW'][attr['idx']],self.vidx['iW'][attr['idx']]] += 2*_lam_cur*attr['yft'].real*attr['yft'].imag
            
            hes[self.vidx['iW'][attr['idx']],self.vidx['U'][attr['f']]] += 2*_lam_cur*attr['yft'].imag*attr['yff'].real
            hes[self.vidx['iW'][attr['idx']],self.vidx['rW'][attr['idx']]] += 2*_lam_cur*attr['yft'].imag*attr['yft'].real
            hes[self.vidx['iW'][attr['idx']],self.vidx['iW'][attr['idx']]] += 2*_lam_cur*attr['yft'].imag*attr['yft'].imag
            
            hes[self.vidx['U'][attr['f']],self.vidx['U'][attr['f']]] += 2*_lam_cur*attr['yff'].imag*attr['yff'].imag
            hes[self.vidx['U'][attr['f']],self.vidx['rW'][attr['idx']]] += 2*_lam_cur*attr['yff'].imag*attr['yft'].imag
            hes[self.vidx['U'][attr['f']],self.vidx['iW'][attr['idx']]] += -2*_lam_cur*attr['yff'].imag*attr['yft'].real
            
            hes[self.vidx['rW'][attr['idx']],self.vidx['U'][attr['f']]] += 2*_lam_cur*attr['yft'].imag*attr['yff'].imag
            hes[self.vidx['rW'][attr['idx']],self.vidx['rW'][attr['idx']]] += 2*_lam_cur*attr['yft'].imag*attr['yft'].imag
            hes[self.vidx['rW'][attr['idx']],self.vidx['iW'][attr['idx']]] += -2*_lam_cur*attr['yft'].imag*attr['yft'].real
            
            hes[self.vidx['iW'][attr['idx']],self.vidx['U'][attr['f']]] += -2*_lam_cur*attr['yft'].real*attr['yff'].imag
            hes[self.vidx['iW'][attr['idx']],self.vidx['rW'][attr['idx']]] += -2*_lam_cur*attr['yft'].real*attr['yft'].imag
            hes[self.vidx['iW'][attr['idx']],self.vidx['iW'][attr['idx']]] += 2*_lam_cur*attr['yft'].real*attr['yft'].real
            
        # to flow limits' contribution to hessian
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            _lam_cur = _lam[idx_tf[brid]]
            
            hes[self.vidx['U'][attr['t']],self.vidx['U'][attr['t']]] += 2*_lam_cur*attr['ytt'].real*attr['ytt'].real
            hes[self.vidx['U'][attr['t']],self.vidx['rW'][attr['idx']]] += 2*_lam_cur*attr['ytt'].real*attr['ytf'].real
            hes[self.vidx['U'][attr['t']],self.vidx['iW'][attr['idx']]] += -2*_lam_cur*attr['ytt'].real*attr['ytf'].imag
            
            hes[self.vidx['rW'][attr['idx']],self.vidx['U'][attr['t']]] += 2*_lam_cur*attr['ytf'].real*attr['ytt'].real
            hes[self.vidx['rW'][attr['idx']],self.vidx['rW'][attr['idx']]] += 2*_lam_cur*attr['ytf'].real*attr['ytf'].real
            hes[self.vidx['rW'][attr['idx']],self.vidx['iW'][attr['idx']]] += -2*_lam_cur*attr['ytf'].real*attr['ytf'].imag
            
            hes[self.vidx['iW'][attr['idx']],self.vidx['U'][attr['t']]] += -2*_lam_cur*attr['ytf'].imag*attr['ytt'].real
            hes[self.vidx['iW'][attr['idx']],self.vidx['rW'][attr['idx']]] += -2*_lam_cur*attr['ytf'].imag*attr['ytf'].real
            hes[self.vidx['iW'][attr['idx']],self.vidx['iW'][attr['idx']]] += 2*_lam_cur*attr['ytf'].imag*attr['ytf'].imag
            
            hes[self.vidx['U'][attr['t']],self.vidx['U'][attr['t']]] += 2*_lam_cur*attr['yff'].imag*attr['ytt'].imag
            hes[self.vidx['U'][attr['t']],self.vidx['rW'][attr['idx']]] += 2*_lam_cur*attr['yff'].imag*attr['ytf'].imag
            hes[self.vidx['U'][attr['t']],self.vidx['iW'][attr['idx']]] += 2*_lam_cur*attr['yff'].imag*attr['ytf'].real
            
            hes[self.vidx['rW'][attr['idx']],self.vidx['U'][attr['t']]] += 2*_lam_cur*attr['ytf'].imag*attr['ytt'].imag
            hes[self.vidx['rW'][attr['idx']],self.vidx['rW'][attr['idx']]] += 2*_lam_cur*attr['ytf'].imag*attr['ytf'].imag
            hes[self.vidx['rW'][attr['idx']],self.vidx['iW'][attr['idx']]] += 2*_lam_cur*attr['ytf'].imag*attr['ytf'].real
            
            hes[self.vidx['iW'][attr['idx']],self.vidx['U'][attr['t']]] += 2*_lam_cur*attr['ytf'].real*attr['ytt'].imag
            hes[self.vidx['iW'][attr['idx']],self.vidx['rW'][attr['idx']]] += 2*_lam_cur*attr['ytf'].real*attr['ytf'].imag
            hes[self.vidx['iW'][attr['idx']],self.vidx['iW'][attr['idx']]] += 2*_lam_cur*attr['ytf'].real*attr['ytf'].real
            
        # SOCR limits' contribution to hessian
        for _,brid in self.branch_list:
            attr = self.brattr[brid]
            _lam_cur = _lam[idx_socr[brid]]
            
            hes[self.vidx['rW'][brid],self.vidx['rW'][brid]] += 2*_lam_cur
            hes[self.vidx['iW'][brid],self.vidx['iW'][brid]] += 2*_lam_cur
            hes[self.vidx['U'][attr['f']],self.vidx['U'][attr['t']]] += -1*_lam_cur
            hes[self.vidx['U'][attr['t']],self.vidx['U'][attr['f']]] += -1*_lam_cur
        
        # ensure hessian is symmetric
        hes = hes + hes.T # add transpose
        idx_diag_hes = np.diag_indices_from(hes)
        hes/= 2. # divide diagonal by factor of 2 which got added due to transpose
        
        # # sparse indices
        # row, col = self.hessianstructure()
        
        return hes
    
    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        pass
    
    def calc_cons_bounds(self):
        
        ub = np.zeros(self.cons_size)
        lb = -LARGE_NUMBER*np.ones(self.cons_size) # proxy for -inf
        
        # set zero lower bound for equalities
        lb[self.cidx['balance_real']] = 0
        lb[self.cidx['balance_reac']] = 0
        
        return ub, lb
        
    def calc_var_bounds(self):
        
        ub = np.ones(self.in_size)
        lb = np.ones(self.in_size)
        
        # voltage squared
        ub[self.vidx['U']] = np.square(self.bus_vmax)
        lb[self.vidx['U']] = np.square(self.bus_vmin)
        
        # real W
        ub[self.vidx['rW']] = +LARGE_NUMBER
        lb[self.vidx['rW']] = -LARGE_NUMBER
        # imag W
        ub[self.vidx['iW']] = +LARGE_NUMBER
        lb[self.vidx['iW']] = -LARGE_NUMBER
        # real generation
        ub[self.vidx['rSg']] = self.gen_pmax
        lb[self.vidx['rSg']] = self.gen_pmin
        # reactive generation'
        ub[self.vidx['iSg']] = self.gen_qmax
        lb[self.vidx['iSg']] = self.gen_qmin
        
        return ub, lb
    
    def calc_x0(self):
        
        # using flat start
        
        x0 = np.zeros(self.in_size)
        # voltage squared
        x0[self.vidx['U']] = 1
        # real W
        x0[self.vidx['rW']] = 1
        # imag W
        x0[self.vidx['iW']] = 0
        # real gen
        x0[self.vidx['rSg']] = 0.5*(self.gen_pmax+self.gen_pmin)
        # react gen
        x0[self.vidx['iSg']] = 0.5*(self.gen_qmax+self.gen_qmin)
        
        return x0