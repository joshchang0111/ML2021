#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Homework 1

# ## Data Preprocessing

# ### Load Data

# In[1]:


import numpy as np
import pandas as pd

data_x_df = pd.read_csv('data_X.csv') ## features
data_t_df = pd.read_csv('data_T.csv') ## labels


# In[2]:


print("total # of data point (features): {}".format(data_x_df.shape))
print("total # of data point (label)   : {}".format(data_t_df.shape))


# In[3]:


data_x_df.head()


# ### Shuffle and Split Data into Training and Validation Set

# In[4]:


def shuffle_feat_label(feats, label):
    idx = np.random.permutation(feats.index)
    feats_random = feats.reindex(idx)
    label_random = label.reindex(idx)
    
    return feats_random, label_random

def split_train_val(feats, label, train_ratio):
    ## Shuffle
    feats, label = shuffle_feat_label(feats, label)
    
    ## Number of data
    num_total = feats.shape[0]
    num_train = int(num_total * train_ratio)
    num_valid = num_total - num_train
    
    ## Split and convert to numpy array
    train_feats, train_label = feats.iloc[0:num_train].to_numpy(), label.iloc[0:num_train].to_numpy()
    valid_feats, valid_label = feats.iloc[num_train:].to_numpy(), label.iloc[num_train:].to_numpy()
    
    return train_feats, train_label, valid_feats, valid_label


# In[5]:


train_x, train_t, valid_x, valid_t = split_train_val(data_x_df, data_t_df, 0.8)
print("# of training   data: {}".format(train_x.shape))
print("# of validation data: {}".format(valid_x.shape))


# ### Normalization

# In[6]:


def normalization(x):
    """Do normalization on each column"""
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    norm_x = (x - mean) / std
    
    return norm_x


# In[7]:


train_x = normalization(train_x)
train_t = normalization(train_t)
valid_x = normalization(valid_x)
valid_t = normalization(valid_t)


# ## Problem 2 Linear Regression

# In[8]:


class RegressionModel:
    """Implementation of Linear Regression Model"""
    
    def __init__(self, num_feat, M, basis="polynomial", regularize=False, _lambda=None):
        
        self.num_feat = num_feat ## number of features
        self.M = M ## order of basis function
        self.basis = basis ## "polynomial", "gaussian"
        self.regularize = regularize ## use regularization or not
        self._lambda = _lambda ## hyperparameter for regularization term
    
    def combinations(self, iterable, r):
        """
        Return the length-r combinations of a list of integers without duplicate ones
        
        Ex: combinations([0, 1, 2], 2) --> (0, 0) (0, 1) (0, 2) (1, 1) (1, 2) (2, 2)
        """
        
        pool = tuple(iterable)
        n = len(pool)
        if not n and r:
            return
        indices = [0] * r
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != n - 1:
                    break
            else:
                return
            indices[i:] = [indices[i] + 1] * (r - i)
            yield tuple(pool[i] for i in indices)
    
    def polynomial_basis(self, x, order):
        """The polynomial basis function"""
        
        for idx, comb in enumerate(list(self.combinations(range(self.num_feat), order))):
            product = np.prod(x[:, comb], axis=1) ## (16346, )
            product = np.expand_dims(product, axis=1) ## (16346, 1)
            
            if idx == 0:
                Phi_i = product
            else:
                Phi_i = np.hstack((Phi_i, product))
                
        return Phi_i
    
    def gaussian_basis(self, x, mu, sigma=0.05):
        """The Gaussian basis function"""
        
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        
    def transform_x(self, x):
        """Convert input with basis function, x->Phi(x)"""
        
        if self.basis.lower() == "polynomial":
            Phi = np.ones((x.shape[0], 1)) ## (16346, 1)
            for order in range(1, self.M + 1):
                Phi_i = self.polynomial_basis(x, order)
                Phi = np.hstack((Phi, Phi_i))
                
        elif self.basis.lower() == "gaussian":
            Phi = np.ones((x.shape[0], 1)) ## (16346, 1)
            for order in range(1, self.M + 1):
                mu = order / (self.M + 1) ## design a mean value for the gaussian distribution
                Phi_i = self.gaussian_basis(x, mu)
                Phi = np.hstack((Phi, Phi_i))
        
        return Phi
    
    def train(self, train_x, train_t):
        """Fit the model on the training set"""
        
        if self.regularize: ## Solution: w = (lambda*I+Phi^T*Phi)^(-1)*Phi^T*t
            Phi = self.transform_x(train_x)
            I = np.eye(Phi.shape[1])
            tmp = np.linalg.inv(self._lambda * I + np.dot(Phi.T, Phi))
            tmp = np.dot(tmp, Phi.T)
            w = np.dot(tmp, train_t)
        
        else: ## Solution: w = (Phi^T*Phi)^(-1)*Phi^T*t
            Phi = self.transform_x(train_x)
            tmp = np.linalg.inv(np.dot(Phi.T, Phi))
            tmp = np.dot(tmp, Phi.T)
            w = np.dot(tmp, train_t)
        
        return w
    
    def eval_rms(self, x, w, t):
        """Evaluate root mean square error"""
        
        Phi = self.transform_x(x)
        y = np.dot(Phi, w)
        rms = np.sqrt(np.mean((y - t) ** 2))
        
        return rms


# ### 2.1 Feature Selection

# #### 2.1.a

# In[9]:


## M = 1 ~ 2
for m in range(1, 3):
    model = RegressionModel(train_x.shape[1], m)
    w = model.train(train_x, train_t)
    
    ## Evaluate RMS error on training set
    train_rms = model.eval_rms(train_x, w, train_t)
    
    ## Evaluate RMS error on validation set
    valid_rms = model.eval_rms(valid_x, w, valid_t)
    
    print("M = {}, train_rms: {:.5f}, valid_rms: {:.5f}".format(m, train_rms, valid_rms))


# #### 2.1.b

# In[10]:


model = RegressionModel(train_x.shape[1], 1)
w = model.train(train_x, train_t)
for idx, wi in enumerate(w.flatten()):
    print("w{} = {}".format(idx, wi))
    
print("\nThe weight with maximum value is w{}".format(np.argmax(w)))


# In[11]:


## Delete one feature at a time and see when which one is removed, the rms error is greatest,
## then the feature is the most contributive one
for i in range(train_x.shape[1]):
    new_train_x = np.concatenate((train_x[:, :i], train_x[:, i + 1:]), axis=1) ## (16346, 7)
    new_valid_x = np.concatenate((valid_x[:, :i], valid_x[:, i + 1:]), axis=1) ## (16346, 7)
    
    model = RegressionModel(new_train_x.shape[1], M=1)
    w = model.train(new_train_x, train_t)
    train_rms = model.eval_rms(new_train_x, w, train_t)
    valid_rms = model.eval_rms(new_valid_x, w, valid_t)
    
    print("Without feature {}, train_rms: {:.5f}, valid_rms: {:.5f}".format(i + 1, train_rms, valid_rms))


# ### 2.2 Maximum Likelihood Approach

# #### 2.2.b

# In[12]:


## M = 1 ~ 20
for m in range(1, 21):
    model = RegressionModel(train_x.shape[1], m, basis="gaussian")
    w = model.train(train_x, train_t)
    
    ## Evaluate RMS error on training set
    train_rms = model.eval_rms(train_x, w, train_t)
    
    ## Evaluate RMS error on validation set
    valid_rms = model.eval_rms(valid_x, w, valid_t)
    
    print("M = {:2d}, train_rms: {:9.5f}, valid_rms: {:13.5f}".format(m, train_rms, valid_rms))


# #### 2.2.c

# In[13]:


def split_n_fold(feats, label, n):
    """Split raw data into n fold, shuffle and do normalization"""
    
    ## Shuffle
    feats, label = shuffle_feat_label(feats, label)
    
    ## Split data into n equal parts
    fold_len = int(feats.shape[0] / n)
    n_parts_x, n_parts_t = [], []
    
    for i in range(n):
        if i == (n - 1):
            start_idx, end_idx = i * fold_len, feats.shape[0]
        else:
            start_idx, end_idx = i * fold_len, (i + 1) * fold_len
        
        part_i_x = feats.iloc[start_idx:end_idx].to_numpy()
        part_i_t = label.iloc[start_idx:end_idx].to_numpy()
        n_parts_x.append(part_i_x)
        n_parts_t.append(part_i_t)
    
    print("{} folds, fold lengths = {}".format(n, [part_i.shape[0] for part_i in n_parts_x]))
    
    ## N folds
    n_folds = []
    for i in range(n):
        ## x
        train_x_i = np.concatenate(n_parts_x[:i] + n_parts_x[i + 1:])
        valid_x_i = n_parts_x[i]
        
        ## t
        train_t_i = np.concatenate(n_parts_t[:i] + n_parts_t[i + 1:])
        valid_t_i = n_parts_t[i]
        
        ## Normalization
        train_x_i = normalization(train_x_i)
        train_t_i = normalization(train_t_i)
        valid_x_i = normalization(valid_x_i)
        valid_t_i = normalization(valid_t_i)
        
        fold_i = [train_x_i, valid_x_i, train_t_i, valid_t_i]
        n_folds.append(fold_i)
    
    return n_folds


# In[14]:


n_folds = split_n_fold(data_x_df, data_t_df, 5)


# In[15]:


for fold_idx, fold in enumerate(n_folds):
    train_x_i, valid_x_i, train_t_i, valid_t_i = fold[0], fold[1], fold[2], fold[3]
    
    for m in range(1, 21):
        ## Train
        model = RegressionModel(train_x.shape[1], M=m, basis="gaussian")
        w = model.train(train_x_i, train_t_i)
        
        ## Evaluate RMS error on training set
        train_rms = model.eval_rms(train_x_i, w, train_t_i)
        
        ## Evaluate RMS error on validation set
        valid_rms = model.eval_rms(valid_x_i, w, valid_t_i)
        
        print("Fold = {}, M = {:2d}, train_rms: {:9.5f}, valid_rms: {:12.5f}".format(fold_idx, m, train_rms, valid_rms))

    print()


# ### 2.3 Maximum A Posterior Approach

# #### 2.3.b

# In[16]:


## M = 1 ~ 20
for m in range(1, 21):
    model = RegressionModel(train_x.shape[1], m, basis="gaussian", regularize=True, _lambda=0.0001)
    w = model.train(train_x, train_t)
    
    ## Evaluate RMS error on training set
    train_rms = model.eval_rms(train_x, w, train_t)
    
    ## Evaluate RMS error on validation set
    valid_rms = model.eval_rms(valid_x, w, valid_t)
    
    print("M = {:2d}, train_rms: {:9.5f}, valid_rms: {:9.5f}".format(m, train_rms, valid_rms))

