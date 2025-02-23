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