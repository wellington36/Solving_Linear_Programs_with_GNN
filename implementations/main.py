import numpy as np
import torch
import random as rd
import torch.nn as nn
import os

from tqdm import tqdm
from model import LPGCN
from generate_data import generate_random_linear_program, solve_linear_program, generate_and_solve_batches
from scipy.optimize import linprog

def train(model, c, A, b, constraint, l, u, sol, feas, out_func, optimizer):
    optimizer.zero_grad()
    out = model(c, A, b, constraint, l, u, out_func)

    loss = nn.MSELoss()
    if out_func == 'feas':
        loss = loss(out, feas)

    elif out_func == 'obj':
        #loss = loss(out, (c[:, None, :] @ sol[:, :, None])[:,0])
        loss = loss(out[:,0], torch.sum(c * sol, dim=1))

    else:
        loss = loss(out, sol)

    loss.backward()
    optimizer.step()
    return loss

def test(model, c, A, b, constraint, l, u, sol, feas, out_func):
    out = model.forward(c, A, b, constraint, l, u, out_func)
    loss = nn.MSELoss()
    if out_func == 'feas':
        loss = loss(out, feas)
    elif out_func == 'obj':
        loss = loss(out[:,0], torch.sum(c * sol, dim=1))
    else:
        loss = loss(out, sol)
    return loss

### PARAMETERS ###
num_constraints = 2
num_variables = 5
batch_size = 10
learning_rate = 0.003
N = 100
out_func = 'feas'

device = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = LPGCN(num_constraints, num_variables).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

data_train = []
for i in range(N):
    c, A, b, constraints, l, u, solution, feasibility = generate_and_solve_batches(batch_size,
                                                                                   num_variables,
                                                                                   num_constraints,
                                                                                   out_func)
    data_train.append([c, A, b, constraints, l, u, solution, feasibility])

init_epochs = 50
epochs = 50
sum_epochs = init_epochs

pbar = tqdm(range(init_epochs))

for epoch in pbar:
    for batch in data_train:
        c, A, b, constraints, l, u, sol, feas = batch
    
        c = c.to(device)
        A = A.to(device)
        b = b.to(device)
        constraints = constraints.to(device)
        l = l.to(device)
        u = u.to(device)
        sol = sol.to(device)
        feas = feas
    
        loss = train(model, c, A, b, constraints, l, u, sol, feas, out_func, optimizer)
        pbar.set_description(f"%.8f" % loss)

torch.save(model.state_dict(), f'gnn_backup/bkp_{out_func}_{sum_epochs}.pt')

for _ in range(10):
    sum_epochs += epochs
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        for batch in data_train:
            c, A, b, constraints, l, u, sol, feas = batch
        
            c = c.to(device)
            A = A.to(device)
            b = b.to(device)
            constraints = constraints.to(device)
            l = l.to(device)
            u = u.to(device)
            sol = sol.to(device)
            feas = feas
        
            loss = train(model, c, A, b, constraints, l, u, sol, feas, out_func, optimizer)
            pbar.set_description(f"%.8f" % loss)
    
    torch.save(model.state_dict(), f'gnn_backup/bkp_{out_func}_{sum_epochs}.pt')
