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

from c2p_model import *


import os 
import argparse 


def train_c2p_model(
    model,optimizer,scheduler,
    training_loader,validation_loader,C_min,C_max,Z_min,Z_max,
    num_epochs,eos,
    training_loss,validation_loss
    ):
    for epoch in range(num_epochs):
        epoch_loss = 0 
        
        for C_data,Z_data in training_loader:
            optimizer.zero_grad()
            
            loss = compute_loss(model,C_data,C_min,C_max,Z_data,Z_min,Z_max,eos)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() 
        epoch_loss /= len(training_loader)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()
        # Compute test loss 
        test_loss = 0
        with torch.no_grad():
            for C_test,Z_test in validation_loader:
                test_loss += compute_loss(model, C_test,C_min,C_max,Z_test,Z_min,Z_max, eos).item()

        test_loss /= len(validation_loader)

        training_loss.append(epoch_loss)
        validation_loss.append(test_loss)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.3e}, Test loss: {test_loss:.3e}")


def log_cosh_loss(y_true, y_pred):
    return torch.mean(torch.log(torch.cosh(y_pred - y_true)))

def compute_loss(model, C,C_min, C_max,  Z,Z_min,Z_max, eos):
    '''
    Eq (C3) of https://arxiv.org/pdf/1306.4953.pdf
    '''
    dtype = torch.float64
    Z_pred = model(C) # Data is normalized inside dataset constructor 
    Z_pred_real = Z_pred * ( Z_max - Z_min ) + Z_min 
    C_real = C * (C_max - C_min) + C_min
    Z_real = Z * (Z_max - Z_min) + Z_min
    htilde = h__z(Z_pred_real,C_real,eos)
    MSEr = log_cosh_loss(Z_pred_real,C_real[:,2].view(-1,1)/htilde).to(dtype)
 
    return log_cosh_loss(Z_pred,Z).to(dtype) + MSEr 

def main(d_ff,model_name, use_fp64=False):
    
    if(os.path.isdir(model_name) ):
        if os.listdir(model_name):
            raise ValueError("Directory {} already exists and is not empty".format(model_name))
    else:
        os.mkdir(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dtype = torch.float64 if use_fp64 else torch.float32
    
    
    # Minkowski metric 
    eta = metric(
    torch.eye(3,device=device), torch.zeros(3,device=device), torch.ones(1,device=device)
    )
    # Gamma = 2 EOS with ideal gas thermal contrib 
    eos = hybrid_eos(100,2,1.8)
    
    # Generate some data
    C, Z = setup_initial_state_meshgrid_cold(eta,eos,200,device,Wmin=1.2,Wmax=1.8)
    # Create a dataset 
    dataset = C2P_Dataset(C, Z, dtype)
    # Set up dataloader 
    batch_size = 32  # You can experiment with this

    # Define split sizes
    train_size = int(0.8 * len(dataset))  # 80% training
    val_size = len(dataset) - train_size  # 20% validation

    # Split dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    err = sanity_check(Z,C,eta,eos)
    print(f"Sanity check error {err:.6e}") 
    
    # Save dataset properties 
    torch.save(dataset.C_max.to(torch.device("cpu")),os.path.join(model_name,'C_max.pt'))
    torch.save(dataset.C_min.to(torch.device("cpu")),os.path.join(model_name,'C_min.pt'))
    torch.save(dataset.Z_max.to(torch.device("cpu")),os.path.join(model_name,'Z_max.pt'))
    torch.save(dataset.Z_min.to(torch.device("cpu")),os.path.join(model_name,'Z_min.pt'))
    
    
    total_epochs = 100000 
    
    
    training_loss = []
    validation_loss = []

    #net = c2p_NN().to(device).to(dtype)
    net = C2PNet(d_ff,dtype).to(device).to(dtype)
    for param in net.parameters():
        if param.grad is not None:
            param.grad = param.grad.to(dtype)

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    for param_group in optimizer.param_groups:
        param_group['params'] = [p.to(dtype) for p in param_group['params']]
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    epochs = total_epochs // len(train_dataloader)
    
    train_c2p_model(net,optimizer,scheduler,train_dataloader,val_dataloader,dataset.C_min,dataset.C_max,dataset.Z_min,dataset.Z_max,epochs,eos,training_loss,validation_loss)

    # Do one more step with LBFGS
    optimizer = torch.optim.LBFGS(net.parameters(), lr=0.01, max_iter=1000)

    def closure():
        optimizer.zero_grad()
        loss = compute_loss(net, train_dataset.dataset.C, dataset.C_min, dataset.C_max, dataset.Z, dataset.Z_min, dataset.Z_max, eos)
        loss.backward()
        training_loss.append(loss.item())
        with torch.no_grad():
            test_loss = compute_loss(net, val_dataset.dataset.C,dataset.C_min, dataset.C_max, dataset.Z, dataset.Z_min, dataset.Z_max, eos).item()
        validation_loss.append(test_loss)
        return loss

    optimizer.step(closure)

    print(f"Final loss: {closure().item()}")  
    
    print(f"Total number of steps {len(training_loss)}, of which {epochs} with Adams.")
    
    fig,ax = plt.subplots()
    ax.semilogy(training_loss,label='Training loss', color="blue")
    ax.semilogy(validation_loss,label='Validation loss', color="red")


    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    #ax.set_ylim(1e-12,1e-1)
    plt.savefig(os.path.join(model_name,"loss_curve.pdf"))
    
    torch.save(net.state_dict(),os.path.join(model_name,"model.pt"))
    
    print("All done!")
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse model parameters.")

    # Integer argument
    parser.add_argument("--d_ff", type=int, required=True, help="Dimension of the hidden layer.")

    # String argument
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model.")

    # Boolean flag argument
    parser.add_argument("--use_fp64", action="store_true", help="Use FP64 if set, otherwise use FP32.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.d_ff, args.model_name, args.use_fp64)
