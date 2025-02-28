# Physics modules
from metric import metric
from hybrid_eos import hybrid_eos

# Numpy and matplotlib
import numpy as np 
import matplotlib.pyplot as plt 

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init 
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np
import matplotlib.pyplot as plt

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # Use 'relu' for SiLU
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # Initialize biases to 0


class C2PNet(nn.Module):
    def __init__(self, d_ff, dtype=torch.float64):
        super(C2PNet, self).__init__()
        self.fc1 = nn.Linear(3, d_ff, dtype=dtype)
        self.fc2 = nn.Linear(d_ff, d_ff, dtype=dtype)
        self.fc3 = nn.Linear(d_ff, d_ff, dtype=dtype)
        self.fc4 = nn.Linear(d_ff+3, 1, dtype=dtype)
        self.activation = nn.SiLU()  # Smooth activation
        self.output = nn.ReLU() # Ensure positivity of z
        self.apply(init_weights)
        
    def forward(self, x):
        identity = x  # Residual connection using one input component
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = torch.cat([x, identity], dim=1) # Residual connection 
        x = self.output(self.fc4(x))
        return x
    
class C2P_Dataset(TensorDataset):
    
    def __init__(self,C,Z,dtype,normalize_data=True):
        
        self.C = C.to(dtype)
        self.Z = Z.to(dtype)
        
        if normalize_data:
            self.normalize()
        
    def normalize(self):
        self.C_max = torch.max(self.C, dim=0, keepdim=True)[0]
        self.C_min = torch.min(self.C, dim=0, keepdim=True)[0]
        self.C = (self.C - self.C_min)/(self.C_max-self.C_min)
        self.Z_max = torch.max(self.Z, dim=0, keepdim=True)[0]
        self.Z_min = torch.min(self.Z, dim=0, keepdim=True)[0]
        self.Z = (self.Z - self.Z_min)/(self.Z_max-self.Z_min)
        
    def __len__(self):
        return self.C.shape[0]
    
    def __getitem__(self,idx):
        return self.C[idx,:], self.Z[idx,:]
    
    
def setup_initial_state_random(metric,eos,N,device,lrhomin=-12,lrhomax=-2.8,ltempmin=-1,ltempmax=2.3,Wmin=1,Wmax=2):
    # Get W, rho and T 
    W = Wmin + (Wmax-Wmin) * torch.rand(N,device=device)
    rho = 10**lrhomin + (10**lrhomax-10**lrhomin) * torch.rand(N,device=device) 
    T = torch.zeros_like(rho,device=device)
    # Call EOS to get press and eps 
    press,eps = eos.press_eps__temp_rho(T,rho)
    # Compute z 
    Z = torch.sqrt(1 - 1/W**2) * W 
    
    # Compute conserved vars 
    sqrtg = metric.sqrtg 
    u0 = W / sqrtg 
    dens = sqrtg * W * rho 
    
    rho0_h = rho * ( 1 + eps ) + press 
    g4uptt = - 1/metric.alp**2 
    Tuptt = rho0_h * u0**2 + press * g4uptt 
    tau = metric.alp**2 * sqrtg * Tuptt - dens 
    
    S = torch.sqrt((W**2-1)) * rho0_h * W
    # Assemble output 
    C = torch.cat((dens.view(-1,1)/metric.sqrtg,tau.view(-1,1)/dens.view(-1,1),S.view(-1,1)/dens.view(-1,1)),dim=1)
    return C, Z.view(-1,1)

def setup_initial_state_meshgrid(metric,eos,N,device,lrhomin=-12,lrhomax=-2.8,ltempmin=-1,ltempmax=2.3,Wmin=1,Wmax=2):
    # Get W, rho and T 
    W = torch.linspace(Wmin,Wmax,N,device=device)
    rho = 10**( torch.linspace(lrhomin,lrhomax,N,device=device) )
    T = 10**( torch.linspace(ltempmin,ltempmax,N,device=device) )
    W, rho, T = torch.meshgrid(W,rho,T, indexing='ij')
    
    W = W.flatten() 
    rho = rho.flatten()
    T = T.flatten() 
    
    # Call EOS to get press and eps 
    press,eps = eos.press_eps__temp_rho(T,rho)
    # Compute z 
    Z = torch.sqrt(1 - 1/W**2) * W 
    
    # Compute conserved vars 
    sqrtg = metric.sqrtg 
    u0 = W / sqrtg 
    dens = sqrtg * W * rho 
    
    rho0_h = rho * ( 1 + eps ) + press 
    g4uptt = - 1/metric.alp**2 
    Tuptt = rho0_h * u0**2 + press * g4uptt 
    tau = metric.alp**2 * sqrtg * Tuptt - dens 
    
    S = torch.sqrt((W**2-1)) * rho0_h * W
    # Assemble output 
    C = torch.cat((dens.view(-1,1)/metric.sqrtg,tau.view(-1,1)/dens.view(-1,1),S.view(-1,1)/dens.view(-1,1)),dim=1)
    return C, Z.view(-1,1)

def setup_initial_state_zerov(metric,eos,N,device,lrhomin=-12,lrhomax=-2.8):
    rho = 10**(torch.linspace(lrhomin,lrhomax,N,device=device))
    
    W = torch.ones_like(rho,device=device) 
    T = torch.zeros_like(rho,device=device)
    
    # Call EOS to get press and eps 
    press,eps = eos.press_eps__temp_rho(T,rho)
    # Compute z 
    Z = torch.sqrt(1 - 1/W**2) * W 
    
    # Compute conserved vars 
    sqrtg = metric.sqrtg 
    u0 = W / sqrtg 
    dens = sqrtg * W * rho 
    
    rho0_h = rho * ( 1 + eps ) + press 
    g4uptt = - 1/metric.alp**2 
    Tuptt = rho0_h * u0**2 + press * g4uptt 
    tau = metric.alp**2 * sqrtg * Tuptt - dens 
    
    S = torch.sqrt((W**2-1)) * rho0_h * W
    # Assemble output 
    C = torch.cat((dens.view(-1,1)/metric.sqrtg,tau.view(-1,1)/dens.view(-1,1),S.view(-1,1)/dens.view(-1,1)),dim=1)
    return C, Z.view(-1,1)

def setup_initial_state_meshgrid_cold(metric,eos,N,device,lrhomin=-12,lrhomax=-2.8,Wmin=1.,Wmax=2):
    # Get W, rho and T 
    W   = torch.linspace(Wmin,Wmax,N,device=device)
    rho = 10**(torch.linspace(lrhomin,lrhomax,N,device=device))
    
    # Meshgrid
    rho, W = torch.meshgrid(rho,W, indexing='ij')
    
    # Flatten
    rho = rho.flatten()
    W = W.flatten() 
    
    # Temperature (0)
    T   = torch.zeros_like(rho,device=device)
    
    # Call EOS to get press and eps 
    press,eps = eos.press_eps__temp_rho(T,rho)
    # Compute z 
    Z = torch.sqrt(1 - 1/W**2) * W 
    
    # Compute conserved vars 
    sqrtg = metric.sqrtg 
    u0 = W / sqrtg 
    dens = sqrtg * W * rho 
    
    rho0_h = rho * ( 1 + eps ) + press 
    g4uptt = - 1/metric.alp**2 
    Tuptt = rho0_h * u0**2 + press * g4uptt 
    tau = metric.alp**2 * sqrtg * Tuptt - dens 
    
    S = torch.sqrt((W**2-1)) * rho0_h * W
    # Assemble output 
    C = torch.cat((dens.view(-1,1)/metric.sqrtg,tau.view(-1,1)/dens.view(-1,1),S.view(-1,1)/dens.view(-1,1)),dim=1)
    return C, Z.view(-1,1)
    
def sanity_check(Z,C, metric, eos):
    t,q,r = torch.split(C,[1,1,1], dim=1)
    htilde = h__z(Z,C,eos)
    
    return torch.mean((Z - r/htilde)**2)

def W__z(z):
    return torch.sqrt(1 + z**2)

def rho__z(z,C):
    return C[:,0].view(-1,1) / W__z(z)

def eps__z(z,C):
    q = C[:,1].view(-1,1)
    r = C[:,2].view(-1,1)
    W = W__z(z)
    return W * q - z * r + z**2/(1+W)

def a__z(z,C,eos):
    eps = eps__z(z,C)
    rho = rho__z(z,C)
    press = eos.press__eps_rho(eps,rho)
    return press/(rho*(1+eps))

def h__z(z,C,eos):
    eps = eps__z(z,C)
    a = a__z(z,C,eos)
    return (1 + eps)*(1+a)