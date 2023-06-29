import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linprog
from tqdm import tqdm

import torch_geometric.nn as G


##########################
# Generate Linear models #
##########################

def generate_random_linear_program(num_variables, num_constraints, bound):
    # Generate random coefficients for the objective function
    c = np.random.rand(num_variables)

    # Generate random coefficients for the constraint matrix
    A = np.random.rand(num_constraints, num_variables)

    # Generate random right-hand side values for the constraints
    b = np.random.rand(num_constraints)

    # Generate random lower and upper bounds for the variables
    lower_bounds = np.ones(num_variables) * -bound
    upper_bounds = np.ones(num_variables) * bound

    # Set the bounds for the variables
    bounds = list(zip(lower_bounds, upper_bounds))

    # Generate random constraint types (0 for <= and 1 for =)
    constraint_types = np.random.choice([0, 1], size=num_constraints, p=[0.7, 0.3])

    # Adjust the constraint matrix and right-hand side based on the constraint types
    for i in range(num_constraints):
        if constraint_types[i] == 1:  # Equality constraint
            A[i] = np.random.choice([0, 1], p=[0.80, 0.20]) * A[i]  # Randomly multiply constraint coefficients by 0 or 1
            b[i] = np.dot(A[i], np.random.rand(num_variables))  # Randomly generate a feasible solution

    # Return the generated linear program
    return c, A, b, constraint_types, bounds


def solve_linear_program(c, A, b, constraint_types, bounds):
    # Solve the linear program
    res = linprog(c, A_ub=A[constraint_types == 0], b_ub=b[constraint_types == 0],
                  A_eq=A[constraint_types == 1], b_eq=b[constraint_types == 1], bounds=bounds)

    print(res)

    # Return the solution
    return res.x, res.status

def generate_and_solve_batches(num_batches, num_variables, num_constraints, out_func):
    batches_c = []
    batches_A = []
    batches_b = []
    batches_constraint_types = []
    batches_lower_bounds = []
    batches_upper_bounds = []
    batches_solutions = []
    batches_feasibility = []
    
    if out_func == 'feas':
        for _ in range(num_batches):
            c, A, b, constraint_types, bounds = generate_random_linear_program(num_variables,
                                                                               num_constraints,
                                                                               float('inf'))

            solution, feasibility = solve_linear_program(c, A, b, constraint_types, bounds)

            lower_bounds, upper_bounds = zip(*bounds)

            batches_c.append(torch.tensor(c, dtype=torch.float32))
            batches_A.append(torch.tensor(A, dtype=torch.float32))
            batches_b.append(torch.tensor(b, dtype=torch.float32))
            batches_constraint_types.append(torch.tensor(constraint_types, dtype=torch.float32))
            batches_lower_bounds.append(torch.tensor(lower_bounds, dtype=torch.float32))
            batches_upper_bounds.append(torch.tensor(upper_bounds, dtype=torch.float32))

            if type(solution) != type(None):
                batches_solutions.append(torch.tensor(solution, dtype=torch.float32))

            else:
                batches_solutions.append(torch.zeros(num_variables, dtype=torch.float32))

            batches_feasibility.append(torch.tensor(1 if feasibility != 2 else 0, dtype=torch.float32))

        return (
            torch.stack(batches_c),
            torch.stack(batches_A),
            torch.stack(batches_b),
            torch.stack(batches_constraint_types),
            torch.stack(batches_lower_bounds),
            torch.stack(batches_upper_bounds),
            torch.stack(batches_solutions),
            torch.stack(batches_feasibility)
        )
    else:
        while len(batches_c) != num_batches:
            c, A, b, constraint_types, bounds = generate_random_linear_program(num_variables,
                                                                               num_constraints,
                                                                              1_000_000)
            solution, feasibility = solve_linear_program(c, A, b, constraint_types, bounds)

            if (type(solution) == type(None)):
                continue
            
            lower_bounds, upper_bounds = zip(*bounds)

            batches_c.append(torch.tensor(c, dtype=torch.float32))
            batches_A.append(torch.tensor(A, dtype=torch.float32))
            batches_b.append(torch.tensor(b, dtype=torch.float32))
            batches_constraint_types.append(torch.tensor(constraint_types, dtype=torch.float32))
            batches_lower_bounds.append(torch.tensor(lower_bounds, dtype=torch.float32))
            batches_upper_bounds.append(torch.tensor(upper_bounds, dtype=torch.float32))
            batches_solutions.append(torch.zeros(num_variables, dtype=torch.float32))
            batches_feasibility.append(torch.tensor(1 if feasibility != 2 else 0, dtype=torch.float32))

        return (
            torch.stack(batches_c),
            torch.stack(batches_A),
            torch.stack(batches_b),
            torch.stack(batches_constraint_types),
            torch.stack(batches_lower_bounds),
            torch.stack(batches_upper_bounds),
            torch.stack(batches_solutions),
            torch.stack(batches_feasibility)
        )


######################
# Generate GNN model #
######################

class LPGNN(nn.Module):
    def __init__(self, num_constraints, num_variables):
        super().__init__()

        self.num_constraints = num_constraints
        self.num_variables = num_variables

        self.num_layers = 5
        ints = np.random.randint(1, 10, size=self.num_layers)
        dims = [2 ** i for i in ints]
        
        # Encode the input features into the embedding space
        self.fv_in = G.MLP([2, 32, dims[0]])
        self.fw_in = G.MLP([3, 32, dims[0]])

        # Hidden states of left nodes
        self.fv = nn.ModuleList([G.MLP([dims[l-1], 32, dims[l]]) for l in range(1, self.num_layers)])
        self.gv = nn.ModuleList([G.MLP([dims[l-1] + dims[l], 32, dims[l]]) for l in range(1, self.num_layers)])

        # Hidden states of right nodes
        self.fw = nn.ModuleList([G.MLP([dims[l-1], 32, dims[l]]) for l in range(1, self.num_layers)])
        self.gw = nn.ModuleList([G.MLP([dims[l-1] + dims[l], 32, dims[l]]) for l in range(1, self.num_layers)])
        
        # Feas and obj output function
        self.f_out = G.MLP([2 * dims[self.num_layers-1], 1, 1])

        # Sol output function
        self.fw_out = G.MLP([3 * dims[self.num_layers-1], 32, 1])

    def construct_graph(self, c, A, b, constraints, l, u):
        hv = torch.cat((b.unsqueeze(2), constraints.unsqueeze(2)), dim=2)
        hw = torch.cat((c.unsqueeze(2), l.unsqueeze(2), u.unsqueeze(2)), dim=2)
        E = A
        return hv, hw, E

    def init_features(self, hv, hw):
        hv_0 = []
        for i in range(self.num_constraints):
            hv_0.append(self.fv_in(hv[:, i]))

        hw_0 = []
        for j in range(self.num_variables):
            hw_0.append(self.fw_in(hw[:, j]))

        hv = torch.stack(hv_0, dim=1)
        hw = torch.stack(hw_0, dim=1)
        return hv, hw

    def layer_left(self, hv, hw, E, layer):
        
        hv_l = []
        batch = hv.shape[0]
        for i in range(self.num_constraints):
            
            s = []
            for j in range(self.num_variables):
                s.append(torch.mul(E[:, i, j, None], self.fw[layer](hw[:, j])))

            s = torch.sum(torch.stack(s, dim=1), dim=1)
            # s = torch.flatten(s, 1)
            # hv_flat = torch.flatten(hv[:, i], 1)

            joint = torch.cat((hv[:, i], s), dim=1)

            hv_l.append(self.gv[layer](joint))
        
        hv_l = torch.stack(hv_l, dim=1)

        return hv_l
    
    def layer_right(self, hv, hw, E, layer):
        
        hw_l = []
        for j in range(self.num_variables):

            s = []
            for i in range(self.num_constraints):
                s.append(torch.mul(E[:, i, j, None], self.fv[layer](hv[:, i])))
            
            s = torch.sum(torch.stack(s, dim=1), dim=1)

            joint = torch.cat((hw[:, j], s), dim=1)
            
            hw_l.append(self.gw[layer](joint))
        
        hw_l = torch.stack(hw_l, dim=1)

        return hw_l
    
    def single_output(self, hv, hw):
        y_out = self.f_out(torch.cat((torch.sum(hv, 1), torch.sum(hw, 1)), dim=1))
        return y_out
    
    def sol_output(self, hv, hw):
        sol = []
        for j in range(self.num_variables):
            joint = torch.cat((torch.sum(hv, 1), torch.sum(hw, 1), hw[:, j]), dim=1)
            sol.append(self.fw_out(joint))

        sol = torch.stack(sol, dim=1)

        return sol[:, :, 0]

    def forward(self, c, A, b, constraints, l, u, phi = 'feas'):

        hv, hw, E = self.construct_graph(c, A, b, constraints, l, u)
        hv, hw = self.init_features(hv, hw)

        for l in range(self.num_layers-1):
            old_hv = hv
            hv = self.layer_left(hv, hw, E, l)
            hw = self.layer_right(old_hv, hw, E, l)

        if phi == 'feas':
            output = self.single_output(hv,hw)
            bins = [1 if elem >= 1/2 else 0 for elem in output]
            return bins
        
        elif phi == 'obj':
            return self.single_output(hv,hw)
        
        elif phi == 'sol':
            return self.sol_output(hv,hw)
        
        else:
            return "Please, choose one type of function: feas, obj or sol"

########################
# Generate Train Data  #
########################

num_constraints = 10
num_variables = 50
batch_size = 2
out_func = 'sol'

data_train = []
for i in range(50):
    c, A, b, constraints, l, u, solution, feasibility = generate_and_solve_batches(batch_size,
                                                                                   num_variables,
                                                                                   num_constraints,
                                                                                   out_func)
    data_train.append([c, A, b, constraints, l, u, solution, feasibility])

def train(model, c, A, b, constraint, l, u, sol, feas, out_func, optimizer):
    optimizer.zero_grad()

    #model = LPGNN(num_constraints, num_variables)

    out = model.forward(c, A, b, constraint, l, u, out_func)

    print(feas, out)
    loss = nn.MSELoss()
    if out_func == 'feas':
        loss = loss(out, feas)
    elif out_func == 'obj':
        loss = loss(out, c.T @ sol)
    else:
        loss = loss(out, sol)
    loss.backward()
    optimizer.step()
    return loss

device = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = LPGNN(num_constraints, num_variables).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

out_func = 'sol'

# training model
epochs = 10
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
        feas = feas.to(device)
        loss = train(model, c, A, b, constraints, l, u, sol, feas, out_func, optimizer)
        pbar.set_description(f"%.8f" % loss)