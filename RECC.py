from GPy.util import linalg
from GPy.util import choleskies, diag
import numpy as np
from scipy.linalg.blas import dgemm, dsymm, dtrmm
from GPy.util.linalg import jitchol, dpotri
from mix import diag_HtKH, dot3rl, dot3lr

from IPython.display import clear_output


# Manuel Schuerch, IDSIA/USI, 24.05.2019


class REC:

    def __init__(self, X, Y, kernel, nEpochs, batchsize, params, params_OPT, \
    params_EST = {'σ0': True, 'ls': True, 'σn': True, 'R': True},\
    α=0.5, PERM=True, Xtest=None, Ftest=None, Ytest=None):

    # params = {'σ0': σ0, 'ls': ls, 'σn': σn, 'R': RR.copy()}


        self.α = α
        self.nEpochs = nEpochs
        self.batchsize = batchsize
        self.N = Y.shape[0]
        self.D = X.shape[1]
        self.M = params['R'].shape[0]
        self.num_inducing = self.M
        self.PERM = PERM

        self.kern = kernel

        self.X_all = X
        self.Y_all = Y

        self.yTest = Ytest
        self.Xtest = Xtest
        self.fTest = Ftest

        self.params = params
        self.params_EST = params_EST

        self.params_optim = params_OPT

        self.nHypers = 1 + self.D*kernel.ARD + (not kernel.ARD)*1 + 1
        self.const_jitter = 1e-7


        self.init_mini_batches()

        # statistics
        self.STATS = np.zeros((4,nEpochs))
        self.STATS_RR = np.zeros((self.M,self.D,nEpochs))
        self.STATS_Y = np.zeros((self.nHypers,nEpochs))


    # compute quantities for mini-batch updates
    def init_mini_batches(self):

        self.nUpdates = int(np.ceil(self.N/self.batchsize))
        div = (self.N+self.batchsize)/self.batchsize
        END = int(np.ceil(div)*self.batchsize)
        self.inds_stops = np.arange(0,END,self.batchsize)

        if self.PERM:
            self.perm = np.random.permutation(self.N)
        else:
            self.perm = np.arange(self.N)

    def new_mini_batch(self, batch_ID):
        # draw new mini-batch and returns global indices
        inds = self.perm[self.inds_stops[batch_ID]:self.inds_stops[batch_ID+1]]
        return inds

    def reset_perm(self):

        # new permutation of indices
        if self.PERM:
            self.perm = np.random.permutation(self.N)


    def reset_epoch(self):

        # update kernel with new hyperparams
        self.kern.lengthscale = self.params['ls'].copy()
        self.kern.variance = self.params['σ0']**2

        σ_n2 = self.params['σn']**2
        Z = self.params['R']


        # initialize all prior quantities
        self.n = np.zeros(self.num_inducing)                    # natural mean vector (num_output = 1!)
        self.P = self.kern.K(Z)                                 # covariance matrix
        diag.add(self.P, self.const_jitter)
        L_P = jitchol(self.P)
        self.C, _ = dpotri(L_P, lower=1)                        # precision matrix
        self._log_marginal_likelihood = 0.0                     # log marginal likelihood
        self._log_Det_C = -2*sum(np.log(np.diag(L_P)))          # log determinant of C

        self.Krr = self.P
        self.iKrr = self.C

        # derivative quantities
        J = self.num_inducing                                   # number of inducing points
        JD = self.num_inducing*self.kern.input_dim              # number of inducing points times dimension
        if self.params_EST['R']:
            self.dn_dR = np.zeros((J,JD))                       # derivative of natural mean wrt inducing inputs (Rjd: R11,...,R1D, R21,...,RJD)
            self.dC_dR = np.zeros((J,J,JD))                     # derivative of precision matrix wrt inducing inputs (Rjd: R11,...,R1D, R21,...,RJD)
            self.dψ_dR = np.zeros((J,self.kern.input_dim))      # gradients of inducing inputs

            dKrr_sparse = self.kern.dK_dX(Z)
            for j in range(0,self.num_inducing):
                for d in range(0,self.kern.input_dim):

                    jd = j*self.kern.input_dim + d
                    self.dC_dR[:,:,jd] = -np.outer(np.dot(self.C,dKrr_sparse[:,j,d]), self.C[:,j] )
                    self.dC_dR[:,:,jd] = self.dC_dR[:,:,jd] + self.dC_dR[:,:,jd].T
        else:
            self.dψ_dR = 0.0
            self.dn_dR = 0.0
            self.dC_dR = 0.0


        dKrr_dσ02 = self.kern.dK_dσ02(Z)
        self.dn_dσ02 = np.zeros(J)
        self.dC_dσ02 = -np.dot(np.dot(self.C,dKrr_dσ02),self.C)
        self.dψ_dσ02 = 0.0

        dKrr_dl = self.kern.dK_dl(Z)
        num_lengthscales = dKrr_dl.shape[2]
        self.dn_dl = np.zeros((J,num_lengthscales))
        self.dC_dl = np.zeros((J,J,num_lengthscales))
        self.dψ_dl = np.zeros(num_lengthscales)
        for d in range(0,num_lengthscales):
            self.dC_dl[:,:,d] = -np.dot(np.dot(self.C,dKrr_dl[:,:,d]),self.C)

        self.dn_dσn2 = np.zeros(J)
        self.dC_dσn2 = np.zeros((J,J))
        self.dψ_dσn2 = 0.0


    def compute_stats(self, i_ep):

        mPred, vPred = self.predict_diag(self.Xtest)

        errF, negLogP, cov = self.calc_err(self.fTest, self.yTest, mPred, vPred, self.params['σn']**2)

        self.STATS[0,i_ep] = self._log_marginal_likelihood
        self.STATS[1,i_ep] = errF
        self.STATS[2,i_ep] = negLogP
        self.STATS[3,i_ep] = cov


    def storeHyp(self, i_ep):
        self.STATS_RR[:,:,i_ep] = self.params['R']
        self.STATS_Y[0,i_ep] = self.params['σ0']
        self.STATS_Y[1:(1+self.D),i_ep] = self.params['ls']
        self.STATS_Y[-1,i_ep] = self.params['σn']



    def calc_err(self, fTest, yTest, m, v, σn2):    # fTest: true function, yTEst: noisy function values, m: predicted mean of function, v: predicted variance, all are vectors!

        σy2 = v + σn2
        σy = np.sqrt( σy2 )
        quant = 1.96

        errF = np.sqrt(np.mean( (fTest-m)**2 ))
        negLogP = np.mean( (fTest-m)**2 / σy2 + np.log(σy2) + np.log(np.pi*2))
        cov = np.mean( (yTest <= m+quant*σy ) * (yTest >= m-quant*σy) )

        return errF, negLogP, cov


    def run(self):

        for i_ep in range(self.nEpochs):

            self.reset_perm()
            self.reset_epoch()

            for i_mini_batch in range(self.nUpdates):

                inds = self.new_mini_batch(i_mini_batch)

                self._log_marginal_likelihood,  self.n, self.m, self.C, self.P, self._log_Det_C, \
                self.dn_dR, self.dC_dR, self.dψ_dR, \
                self.dn_dσ02, self.dC_dσ02, self.dψ_dσ02, \
                self.dn_dl, self.dC_dl, self.dψ_dl, \
                self.dn_dσn2, self.dC_dσn2, self.dψ_dσn2 = self.inference(
                    self.n, self.C, self.P, self._log_marginal_likelihood, self._log_Det_C, self.dn_dR, self.dC_dR, self.dψ_dR,
                    self.dn_dσ02, self.dC_dσ02, self.dψ_dσ02, self.dn_dl, self.dC_dl, self.dψ_dl, self.dn_dσn2, self.dC_dσn2, self.dψ_dσn2,
                    self.X_all[inds,:], self.Y_all[inds,:])


                self.update_params()

            self.storeHyp(i_ep)
            self.compute_stats(i_ep)

            clear_output(wait=True)
            print('epoch: ',i_ep)
            print('log marginal likelihood: ',self._log_marginal_likelihood[0])
            print('RMSE ',self.STATS[1,max(i_ep-1,0)],' negLogP ',self.STATS[2,max(i_ep-1,0)],' COV ',self.STATS[3,max(i_ep-1,0)])

            print('σ20: ',self.params['σ0']**2, ' ls: ',self.params['ls'] , ' σ2n: ',self.params['σn']**2 )




    def inference(self, n0, C0, P0, log_marginal_likelihood0, log_Det_C0, dn_dR, dC_dR, dψ_dR, dn_dσ02, dC_dσ02, dψ_dσ02, dn_dl, dC_dl, dψ_dl, dn_dσn2, dC_dσn2, dψ_dσn2, X, Y):

        α = self.α
        α_const = (1-α)/α


        num_data, _ = Y.shape
        num_inducing = n0.shape[0]  # it only works with num_outputs = 1


        y = Y[:,0]   # it only works with num_outputs = 1

        # update kernel with new hyperparams
        self.kern.lengthscale = self.params['ls'].copy()
        self.kern.variance = self.params['σ0']**2

        σ_n2 = self.params['σn']**2
        Z = self.params['R']

        # compute kernel quantities
        Krr = self.kern.K(Z)                                # kernel matrix of inducing inputs
        diag.add(Krr, self.const_jitter)                    # add some jitter for stability reasons
        Kxr = self.kern.K(X,Z)                              # kernel matrix between mini-batch and inducing inputs
        kxx = self.kern.Kdiag(X)  #+const_jitter            # diagonal of kernel matrix auf mini-batch
        L_K = jitchol(Krr)                                  # lower cholesky matrix of kernel matrix
        iKrr, _ = dpotri(L_K)                               # inverse of kernel matrix of inducinv inputs

        self.Krr = Krr
        self.iKrr = iKrr



        # compute state space matrices (and temporary matrices)
        H = np.dot(Kxr,iKrr)                            # observation matrix
        Ht = H.T                                        # transpose of observation matrix
        d = kxx - np.sum(H*Kxr,1)                       # diagonal of correction matrix
        v = α*d + σ_n2                                  # diagonal of actual noise matrix
        a = α_const * ( np.sum(np.log(v)) - num_data*np.log(σ_n2) ) # PEP correction term in marignal likelihoo

        A_ = Ht/v
        α_ = np.dot(P0,n0)

        r = y - np.dot(H,α_)

        # update natural mean and precision + inversion yielding covariance matrix
        # n1 = ns + np.dot(A_,y)
        # C1 = Cs + np.dot(A_,H)

        n1 = n0 + np.dot(A_,y)
        C1 = C0 + np.dot(A_,H)
        L_C = jitchol(C1)
        P1, _ = dpotri(L_C)

        # more temporary matrices
        B_ = np.dot(H,P1)   # iV * H * Li'     # LAPACK?
        β_ = r/v
        γ_ = np.dot(B_.T,β_)
        δ_ = β_ - np.dot(A_.T,γ_)

        # update marginal log likelihood
        log_Det_C1 = 2*sum(np.log(np.diag(L_C)))
        log_Ddet_V =  sum(np.log(v))
        Δ0 = num_data*np.log(2*np.pi) + log_Det_C1 - log_Det_C0 + log_Ddet_V +  np.sum(r*δ_) + a
        log_marginal_likelihood1 = log_marginal_likelihood0 - 0.5*Δ0

        # print('lik_i '+str(0.5*Δ0))


        # compute constant derivatives of likelihood wrt kernel matrices
        dL_dH = 2*( (B_.T/v).T - np.outer(δ_,α_+γ_) )
        dL_dv = - ( np.sum(H*B_,1) -v/α + (r-np.dot(H,γ_))**2 ) / (v**2)

        D_ = α*(Ht*dL_dv).T
        E_ = np.dot(dL_dH,iKrr)

        dL_dKxr = E_ - 2*D_
        dL_dKrr = -np.dot(Ht,E_-D_)

        dL_dkxx = α * dL_dv

        dL_dn = -2*np.dot(P0,np.dot(Ht,δ_) )
        dL_dC = P1 - P0 - np.outer(dL_dn,α_) + np.outer(γ_,γ_)

        # dL_d_dn = 2*σ_n2 *sum(dL_dv) -2*num_data*α_const # wrt to dn
        dL_d_dn = sum(dL_dv) - num_data*α_const/σ_n2   # wrt to σn2

        iVy = y/v
        dH = np.zeros((num_data, num_inducing))

        scaleFact = 1 ###


        if self.params_EST['R']:
            # compute sparse kernel derivatives
            # dKrr_sparse = np.zeros((J,J,D))
            dKrr_sparse = self.kern.dK_dX(Z)#, dK_dR=dKrr_sparse)
            # dKxr_sparse = np.zeros((B,J,D))
            dKxr_sparse = self.kern.dK_dX(X, Z)#, dK_dR=dKxr_sparse)

            # loop over all inducing points
            for j in range(0,num_inducing):
                for d in range(0,self.D):

                    jd = j*self.D + d
                    kjd = dKrr_sparse[:,j,d]
                    k2jd = dKxr_sparse[:,j,d]

                    #dψ_dR[j,d] = dψ_dR[j,d] -0.5*( np.sum(dL_dKrr[:,j]*kjd) + np.sum(dL_dKrr[j,:]*kjd) + np.sum(dL_dKxr[:,j]*k2jd) + np.sum( dL_dn*dn_dR[:,jd]) + np.sum( dL_dC*dC_dR[:,:,jd]) )
                    ### dψ_dR[j,d] = dψ_dR[j,d] -0.5*( np.sum(dL_dkxx *dKxx_diag) +  dL_d_dn   )

                    delta = -0.5*( np.sum(dL_dKrr[:,j]*kjd) + np.sum(dL_dKrr[j,:]*kjd) + np.sum(dL_dKxr[:,j]*k2jd) + np.sum( dL_dn*dn_dR[:,jd]) + np.sum( dL_dC*dC_dR[:,:,jd]) )
                    dψ_dR[j,d] =   delta*scaleFact

                    dH = -np.outer( H[:,j], kjd)
                    dH[:,j] += -np.dot(H,kjd)  + k2jd
                    dH = np.dot(dH,iKrr)

                    dd =  - np.sum(dH * Kxr,1) - H[:,j] * k2jd    #### dKxx_diag for theta!!
                    div = - α*dd / (v**2)
                    dn_dR[:,jd] = dn_dR[:,jd] + np.dot(dH.T,iVy )    + np.dot(Ht,div * y)
                    F_ = np.dot(A_,dH)
                    dC_dR[:,:,jd] = dC_dR[:,:,jd] + F_ + F_.T  + np.dot(Ht * div, H)


        # compute kernel derivatives wrt variance_0
        dKrr_dσ02 = self.kern.dK_dσ02(Z)
        dKxr_dσ02 = self.kern.dK_dσ02(X,Z)
        dkxx_dσ02 = self.kern.dK_dσ02_diag(X)


        # dψ_dσ02 = dψ_dσ02 - 0.5*( np.sum(dL_dKrr*dKrr_dσ02) + np.sum(dL_dKxr*dKxr_dσ02) + np.sum( dL_dn*dn_dσ02) + np.sum( dL_dC*dC_dσ02) )
        # dψ_dσ02 = dψ_dσ02 - 0.5* np.sum(dL_dkxx *dkxx_dσ02)

        delta = - 0.5*( np.sum(dL_dKrr*dKrr_dσ02) + np.sum(dL_dKxr*dKxr_dσ02) + np.sum( dL_dn*dn_dσ02) + np.sum( dL_dC*dC_dσ02) )
        delta = delta - 0.5* np.sum(dL_dkxx *dkxx_dσ02)


        dψ_dσ02 =   delta*scaleFact


        dH = dKxr_dσ02 -np.dot( H, dKrr_dσ02)
        dH = np.dot(dH,iKrr)

        dd =  dkxx_dσ02 - np.sum(dH * Kxr,1)  - np.sum(H * dKxr_dσ02,1)
        div = - α*dd / (v**2)
        dn_dσ02= dn_dσ02 + np.dot(dH.T,iVy )    + np.dot(Ht,div * y)
        F_ = np.dot(A_,dH)
        dC_dσ02 = dC_dσ02 + F_ + F_.T  + np.dot(Ht * div, H)



        # compute kernel derivatives wrt lengthsacle(s)
        dKrr_dl = self.kern.dK_dl(Z)
        dKxr_dl = self.kern.dK_dl(X,Z)
        # dkxx_dl = kern.dK_dl_diag(X)   # zero anyway

        # loop over all lengthscales
        num_lengthscales = dKrr_dl.shape[2]
        for d in range(0,num_lengthscales):

            delta = - 0.5*( np.sum(dL_dKrr*dKrr_dl[:,:,d]) + np.sum(dL_dKxr*dKxr_dl[:,:,d])  + np.sum( dL_dn*dn_dl[:,d]) + np.sum( dL_dC*dC_dl[:,:,d]) )
            #############################

            dψ_dl[d] =   delta*scaleFact
            dH = dKxr_dl[:,:,d] -np.dot( H, dKrr_dl[:,:,d])
            dH = np.dot(dH,iKrr)

            dd =  - np.sum(dH * Kxr,1)  - np.sum(H * dKxr_dl[:,:,d],1)
            div = - α*dd / (v**2)
            dn_dl[:,d]= dn_dl[:,d] + np.dot(dH.T,iVy )    + np.dot(Ht,div * y)
            F_ = np.dot(A_,dH)
            dC_dl[:,:,d] = dC_dl[:,:,d] + F_ + F_.T  + np.dot(Ht * div, H)


        # gaussian noise variance
        delta = - 0.5*( np.sum( dL_dn*dn_dσn2 ) + np.sum( dL_dC*dC_dσn2 ) +  dL_d_dn  )
        # dψ_dσn2 = dψ_dσn2

        dψ_dσn2 =   delta*scaleFact

        div = - 1.0 / (v**2)
        dn_dσn2= dn_dσn2   + np.dot(Ht,div * y)
        dC_dσn2  = dC_dσn2   + np.dot(Ht * div, H)



        m1 = np.dot(P1,n1)

        return log_marginal_likelihood1, n1, m1, C1, P1, log_Det_C1, dn_dR, dC_dR, dψ_dR,  dn_dσ02, dC_dσ02, dψ_dσ02, dn_dl, dC_dl, dψ_dl,  dn_dσn2, dC_dσn2, dψ_dσn2



    def predict_diag(self, Xtest, NOISE=True):
        # returns the mean and covariance (only diag) for test points (with/without likelihood noise)

        Kxx_diag = self.kern.Kdiag(Xtest) + self.const_jitter
        Krx = self.kern.K(self.params['R'], Xtest)

        H = np.dot(self.iKrr, Krx)

        # predicted mean and variance of GP
        m = np.dot(H.T, self.m)
        v = Kxx_diag - np.sum( H * Krx, 0) + diag_HtKH(H, self.P)

        if NOISE:
            v += self.params['σn']**2

        return m, v


    def update_params(self):

        #Ei and dψ are values and derivatives for the approximate marginal likelihood which we want to minimize
        # therefore we have to provide the negative gradients -dψ

        #self
        #dψ_dR, dψ_dσ02, dψ_dl, dψ_dσn2

        # transformation of derivatives
        # since GPy computes the derivatives wrt l and σ02, we have to transfrom the derivatives in log scale
        # s_0 = log(σ0), thus: dQ_ds_0 = dQ_dσ02 * dσ02_ds_0 = dQ_dσ02 * 2 * σ02
        # s_l = log(l), thus: dQ_ds_l = dQ_dl * dl_ds_l = dQ_dl * l

        dEi_Y_trans = np.zeros(self.nHypers)
        dEi_Y_trans[0] = self.dψ_dσ02 * 2 * self.params['σ0']**2
        dEi_Y_trans[1:-1] = self.dψ_dl * self.params['ls']
        dEi_Y_trans[-1] = self.dψ_dσn2 * 2 * self.params['σn']**2


        if self.params_EST['ls']:
            self.params['ls'] = np.exp( self.params_optim['ls'].update(np.log(self.params['ls']), -dEi_Y_trans[1:-1]) )

        if self.params_EST['σ0']:
            self.params['σ0'] = np.exp( self.params_optim['σ0'].update(np.log(self.params['σ0']), -dEi_Y_trans[0]) )

        if self.params_EST['σn']:
            self.params['σn'] = np.exp( self.params_optim['σn'].update(np.log(self.params['σn']), -dEi_Y_trans[-1]) )

        if self.params_EST['R']:
            self.params['R'] = self.params_optim['R'].update( self.params['R'], -self.dψ_dR)
