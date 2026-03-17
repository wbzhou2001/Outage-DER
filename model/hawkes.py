import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import arrow
import os
from .discretizer import MarkDiscretizer

class ExponentialMultivariateKernel(torch.nn.Module):

    def __init__(self, alpha, beta):
        '''
        Args:
        - alpha:    [ S, S ] np
        - beta:     [ 1 ] np
        '''
        super().__init__()
        self.alpha  = torch.nn.Parameter(torch.tensor(alpha).float())   # [ S ] torch
        self.beta   = torch.nn.Parameter(torch.tensor(beta).float())    # [ 1 ] torch

    def forward(self, x, xp):
        '''
        Args:
        - both  [ batch_size, 2 ] torch
        Returns:
        - val:  [ batch_size ] torch
        '''
        alpha_ = self.alpha[xp[:, 1].long(), x[:, 1].long()]  # [ batch_size ] torch  
        val    = alpha_ * self.beta * torch.exp( - self.beta * torch.abs(x[:, 0] - xp[:, 0]))   # [ batch_size ]
        return val
    
    @staticmethod
    def top_k_neighbors(k):
        # TODO: select the top k nearst neightbor to be included in initialized alpha
        raise NotImplementedError()

class OutageHawkes(torch.nn.Module):
    '''
    Hawkes model with
    - continuous time
    - discrete space
    - continuous marks
    Specialized for outage occurence modeling
    '''
    def __init__(self, T, S, M,
                 cov_tr, cov_te,
                 kernel_kwds,
                 int_res    = 100, mark_res = 10,
                 hist_clip  = 1024, 
                 verbose = True):
        '''
        Args:
        - T:        time range, e.g. [0., 1.]
        - S:        int, the number of spatial units, e.g., 50
        - M:        mark space bounds (n_mark, 2), e.g., [[0., 1.], [0., 1.]] 
        - mark_res: resolution of mark discretization
        - cov_tr:   [ S, n_tr, n_cov ] np
        - cov_te:   [ S, n_te, n_cov ] np
        '''
        # -----------
        # Basics
        # -----------
        super().__init__()
        self.T =  T 
        self.M =  np.array(M)   # [ n_mark, 2 ]
        self.S =  S
        self.int_res    = int_res
        self.mark_res   = mark_res 
        self.verbose    = verbose
        self.hist_clip  = hist_clip
        self.kernel     = ExponentialMultivariateKernel(**kernel_kwds)
        if verbose:
            print(f'History clipper set to {hist_clip}')
            print(f'Mark resolution set to {mark_res}')
            print(f'Integral resolution set to {int_res}')
        # -----------
        # Parameters: mark conditional PDF
        # -----------
        n_mark = self.M.shape[0]
        self.md         = MarkDiscretizer(M, mark_res, clip=True)
        self.fc         = nn.Linear(2, mark_res**n_mark)
        self.softmax    = nn.Softmax(dim = -1)
        self.mark_bias  = nn.Parameter(torch.zeros(mark_res**n_mark).float(), requires_grad=False)  # [ mark_res**M ] torch
        # naive initialization
        nn.init.constant_(self.fc.weight, 0.0)  # all weights = 0
        nn.init.constant_(self.fc.bias, 0.0)    # all biases = 0
        # -----------
        # Parameters: covariate
        # -----------
        self.cov_tr = torch.tensor(cov_tr).float() if isinstance(cov_tr, np.ndarray) else cov_tr # [ S, n_time_tr, n_cov ] np
        self.cov_te = torch.tensor(cov_te).float() if isinstance(cov_te, np.ndarray) else cov_te # [ S, n_time_te, n_cov ] np
        self.cov    = torch.cat([self.cov_tr, self.cov_te], 1)  # [ S, n_time_tr + n_time_te , n_cov ] torch
        # naive initialization (zero matrix + frequency base rate)
        self.coef_mat = torch.nn.Parameter(torch.zeros(self.cov_tr.shape[-1]))  # [ n_cov ]
        self.mu_ = nn.Parameter(torch.ones(self.S).float(), requires_grad=False)    # [ S ]

    def fit(self, data, num_epochs, lr, save_folder, patience = 5):
        '''
        Args:
        - data:         [ seq_len, data_dim = 2 + n_mark ] numpy or torch
        Returns:
        - model's state dict
        - model's training log
        '''
        data            = torch.tensor(data).float() if isinstance(data, np.ndarray) else data  # [ seq_len, data_dim ] torch
        opt             = optim.Adadelta(self.parameters(), lr=lr)
        # Initializing self.mark_bias
        indices     = self.md.transform(data[:, 2:].detach().numpy())           # [ seq_len ] np
        mask        = np.arange(self.mark_res**self.M.shape[1])[:, None] == indices[None, :]    # [ int_res**M, seq_len ] np
        count       = torch.tensor(mask.sum(1)).float()                                         # [ int_res**M ] torch
        self.mark_bias.data = count / max(count.sum(), 1.)                      # [ int_res**M ] torch
        # Init self.mu_
        y = np.bincount(data[:, 1].numpy().astype(int), minlength = self.S)     # [ n_S ] np
        y = torch.tensor(y).float() / (self.T[1] - self.T[0])                   # [ n_S ] torch, average count across substations
        self.mu_.data = y                                                       # [ n_S ] torch
        # Discretize data mark
        indices = self.md.transform(data[:, 2:])    # [ seq_len ]
        data    = torch.cat([data[:, :2], torch.tensor(indices.reshape(-1, 1)).float()], dim=1) # [ seq_len, 3 ] torch    
        # ---------------
        # Training
        # ---------------
        loss_list = []
        for i in range(num_epochs):
            opt.zero_grad()
            loglik      = self.loglik(data)
            loss        = - loglik.mean()   # maximize loglikelihood
            loss.backward()
            opt.step()
            loss_list.append(loss.item())

            if i % (max(num_epochs // 10, 1)) == 0 and self.verbose:
                print(f"[{arrow.now()}] Epoch : {i} \t Loss : {loss_list[-1]:.5e}")

            if np.argmin(loss_list) < len(loss_list) - patience:
                print('Early stopping.')
                break

        # ---------------
        # Save trained result
        # ---------------
        os.makedirs(save_folder, exist_ok=True)
        torch.save(self.state_dict(), save_folder + '/state_dict.pth')
        np.save(save_folder + '/loss_list.npy', np.array(loss_list))
        print('Model has been trained') if self.verbose else None

    def load(self, save_folder):
        self.load_state_dict(torch.load(save_folder + '/state_dict.pth'))
        print('Model has been loaded') if self.verbose else None

    def ground_lam(self, x, h):
        '''
        Conditional intensity function of the ground process
        Args:
        - x   : [ batch_size, st_dim = 2 ] torch or torch scalar
        - h   : [ batch_size, his_len, st_dim = 2 ] torch
        Returns:
        - lam : [ batch_size ] torch  
        '''
        mu      = self.mu(x)    # [ batch_size ] torch
        shape   = h.shape 
        if shape[0] == 0 or shape[1] == 0 or shape[2] == 0:
            return mu           #  if no history, return baserate
        # pariwise kernel
        x_ext   = x.unsqueeze(1).repeat(1, shape[1], 1)                             # [ batch_size, his_len, 2 ] torch
        x_batch, h_batch = x_ext.reshape(-1, shape[2]), h.reshape(-1, shape[2])     # both [ ext_batch_size, 2 ] torch
        K       = self.kernel(x_batch, h_batch)                                     # [ ext_batch_size ]
        K       = K.reshape(shape[0], shape[1])                                     # [ batch_size, his_len ]
        # Mask all paddings (value=-1)
        mask    = h[:, :, 0] != -1.     # [ batch_size, his_len ] torch
        K       = K * mask              # [ batch_size, his_len ] torch 
        lam     = mu + K.sum(-1)        # [ batch size ] torch
        lam     = torch.clamp(lam, min=1e-5, max=None)  # [ batch_size ] torch, relu to avoid negative values
        return lam

    def lam(self, x, h):
        '''
        Conditional intensity function
        Args:
        - x   : [ batch_size, 3 ] torch or torch scalar
        - h   : [ batch_size, his_len, 3 ] torch
        Returns:
        - lam : [ batch_size ] torch  
        '''
        glam    = self.ground_lam(x[:, :2], h[:, :, :2])        # [ batch_size ]
        prob    = self.mark_prob(x[:, :2])                      # [ batch_size, mark_res**M ] torch
        indices = x[:, 2].long()                                # [ batch_size ] torch
        prob    = torch.gather(prob, 1, indices.unsqueeze(1)).squeeze(1)    # [ batch_size ] torch, same as using prob[:, indices].diag() but more efficient
        lam     = glam * prob                           # [ batch_size ]
        lam     = torch.clamp(lam, min=1e-5, max=None)  # [ batch_size ]
        return lam       

    def mark_prob(self, st):
        '''
        Args:
        - st   : [ batch_size, 2 ] torch or torch scalar
        Returns:
        - prob: [ batch_size, mark_res**M ] torch
        '''
        a = self.fc(st)                         # [ batch_size, mark_res**M ] torch
        a = self.softmax(a + torch.log(self.mark_bias))    # [ batch_size, mark_res**M ] torch
        return a
    
    def simulate(self, data, t_start, t_end, lam_bar):
        '''
        Joint thinning algorithm (i.e., not Ogata's thinning, more flexible but less efficiency)
        Args:
        - data:         [ seq_len, data_dim ] np, history data 
        Returns:
        - sim_traj:     [ seq_len, data_dim ] numpy, simulated trajectory
        '''
        self.eval()
        # TODO: only keep the first 1024 data points, longer sequences have little effect and slows down the process
        data    = torch.tensor(data[-self.hist_clip:]).float() if isinstance(data, np.ndarray) else data # [ seq_len, data_dim ] torch
        retained_data  = [ x for x in data[:, :2] ] # [ seq_len, 2 ] torch
        # ---------------
        # Ground process simulation: homogenous spatio-temporal poisson process)
        # ---------------
        X = np.array([[t_start, t_end]] + [[0., self.S]])               # [ st_dim = 2, 2 ]
        N = np.random.poisson(np.diff(X, 1).prod() * lam_bar)           # scalar
        raw_data = np.random.uniform(X[:, 0], X[:, 1], size=(N, 2))     # [ N, 2 ]
        raw_data = raw_data[raw_data[:, 0].argsort(), :]                # [ N, 2 ]
        raw_data = torch.tensor(raw_data).float()
        raw_data[:, 1] = torch.floor(raw_data[:, 1])                    # convert continuous space info to discrete index
        # ---------------
        # Rejection phase
        # ---------------
        lam_list = []
        for x in raw_data:
            h = torch.stack(retained_data)  # [ seq_len, 2 ]
            lam = self.ground_lam(x.unsqueeze(0), h.unsqueeze(0))
            lam_list.append(lam.item())
            D = np.random.uniform()

            if lam <= lam_bar:
                if lam >= lam_bar * D:
                    retained_data.append(x)
                else:
                    pass # do not retain, move to next
            else:
                print(f'Exceeded maximum lambda! {lam.detach().numpy().item() : .2f} > {lam_bar : .2f}') if self.verbose else None
                raise NotImplementedError
        # ---------------
        # Organize output
        # ---------------
        retained_data = retained_data[len(data):]
        if len(retained_data) == 0:
            print('No points retained!')
            retained_data = np.zeros((0, data.shape[-1]))    # [ 0, 2 + n_mark ] torch
        else:
            retained_data = torch.stack(retained_data)          # [ pred_len, 2 ] torch
            # ---------------
            # Classify for the marks (maximum porbability)
            # ---------------
            prob = self.mark_prob(retained_data)                    # [ pred_len, mark_res**M ] torch
            # indices = torch.argmax(prob, dim=1).detach().numpy()    # [ pred_len ] np, maximum liklihood
            indices = torch.multinomial(prob, num_samples=1).squeeze(1).detach().cpu().numpy() # [ pred_len ] np, sampling
            marks = self.md.inverse_transform_to_bin_centers(indices)       # [ pred_len, n_marks ] np
            retained_data = np.concatenate([retained_data.detach().numpy(), marks], axis=1) # [ pred_len, 2 + n_marks ] np

        print(f'Maximum lambda: {np.max(lam_list) : .2f}, Lam_bar: {lam_bar : .2f}') if self.verbose else None
        return retained_data    # [ pred_len, data_dim ] np

    def loglik(self, data):
        '''
        Args:
        - data      : [ seq_len, 3 ] torch
        Returns:
        - loglik    : torch float
        '''
        xs, hs = [], []
        for i in range(len(data)):
            x, h = data[i], data[:i] # [ 3 ] and [ his_len, 3 ] torch
            # Use -1 as padding
            h = torch.nn.functional.pad(h, pad=(0, 0, 0, len(data)-len(h)), value=-1.)   # [ max_his_len, 3 ]
            xs.append(x)
            hs.append(h)
        xs = torch.stack(xs, 0)   # [ seq_len, 3 ]
        hs = torch.stack(hs, 0)             # [ seq_len, max_his_len, 3 ]
        hs = hs[:, -self.hist_clip:, :]     # [ seq_len, hist_clip, 3 ]
        lams = self.lam(xs, hs)             # [ seq_len ] torch
        # See Reinhart's review of point process Eq.(8)
        term1 = torch.log(lams).sum()                                       # scalar torch
        term2 = self.integrand(data).sum() * self.T[-1] / self.int_res      # scalar torch
        loglik = term1 - term2
        return loglik                           # scalar torch

    def integrand(self, data,
                    T = None, int_res = None):
        '''
        Args:
        - T                 : list of starting time and ending time, e.g. [ 0., 30. ], default uses the same as the training data
        - data              : [ number of datapoints, 2 ] np or torch
        Returns:
        - lams              : [ int_res, S, mark_res**M ] torch
        '''
        # ------------
        # Load parameters
        # ------------
        if T is None:
            T = self.T
        if int_res is None:
            int_res = self.int_res
        data = torch.tensor(data).float() if isinstance(data, np.ndarray) else data # [ seq_len, 2 ] torch
        # ------------
        # Get meshgrid locations
        # ------------
        tt  = np.linspace(T[0], T[1], int_res)  # [ int_res ] np
        ss  = np.arange(self.S)                 # [ S ] np
        mm  = np.arange(self.mark_res**self.M.shape[0]) # [ mark_res**M ] np
        axes = [tt, ss, mm] # ragged list collection
        xx  = np.meshgrid(*axes, indexing='ij')
        xx  = np.stack([x.reshape(-1) for x in xx], axis=-1)        # shape [ prod_i ni, 3 ], all the points in X
        # ------------
        # Collect pairwise data 
        # ------------
        hs = []
        for x in xx:
            mask    = data[:, 0] < x[0]
            h       = data[mask]
            h       = torch.nn.functional.pad(h, pad=(0, 0, 0, len(data)-len(h)), value=-1.)   # [ max_his_len, 2 ]
            hs.append(h)
        xx  = torch.tensor(xx).float()  # [ prod_i ni, 3 ]
        hh  = torch.stack(hs, 0)        # [ prod_i ni, max_his_len, 3 ]
        hh  = hh[:, -self.hist_clip:, :]    # [ seq_len, hist_clip, 3 ]
        lams = self.lam(xx, hh)         # [ prod_i ni ]
        lams = lams.reshape(*[len(axes[i]) for i in range(len(axes))]) # [ int_res, S, mark_res**M ] torch
        return lams
    
    def forward(self, x):
        '''[ batch_size, seq_len, data_dim ] torch'''
        return self.loglik(x)   # return conditional intensities and corresponding log-likelihood
    
    def mu(self, x):
        '''
        Args:
        - x:    [ batch_size, data_dim ] torch
        Returns:
        - [ batch_size ] torch
        '''
        t_index = torch.floor(self.cov_tr.shape[1] / (self.T[1] - self.T[0]) * x[:, 0]).long()  # [ batch_size ] torch, time index of the datapoint
        t_index = torch.clamp(t_index, min = None, max = self.cov.shape[1] - 1).long()
        s_index = x[:, 1].long()                # [ batch_size ] torch
        cov     = self.cov[s_index, t_index]    # [ batch_size, cov ] torch
        mu_     = self.mu_[s_index] + cov @ self.coef_mat   # [ batch_size ] torch
        mu_     = mu_ + mu_ * 0.1 * (x[:, 0] - self.T[0]) / (self.T[1] - self.T[0]) # TODO: assume a linear increasing trend of the baserate of event occurence
        return mu_