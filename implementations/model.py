import numpy as np
import torch
import torch_geometric.nn as G
import torch.nn as nn

class LPGCN(nn.Module):
    ''' 
    A GNN model for solving linear programs. The LP is modeled as a weighted bipartite graph with constraints on the left and
    variables on the right. The optimization problems are of the form: min c^T x s.t. Ax (constraints) b, l <= x <= u
    ------------------------------------------------------------------------------------------------------------------

    Attributes:
        num_constraints : int
            number of constraints in the LP

        num_variables : int
            number of variables in the LP

        num_layers : int
            number of layers in the GNN
    ------------------------------------------------------------------------------------------------------------------
    '''

    def __init__(self, num_constraints, num_variables, num_layers=5):
        '''
        Parameters
        ------------------------------------------------------------------------------------------------------------------
        num_constraints : int
            number of constraints in the LP

        num_variables : int
            number of variables in the LP

        num_layers : int
            number of layers in the GNN
        ------------------------------------------------------------------------------------------------------------------
        '''

        super().__init__()

        self.num_constraints = num_constraints
        self.num_variables = num_variables

        self.num_layers = num_layers

        # Generate random integers for the dimensions of the hidden layers.
        # The dimensions are powers of 2, with min = 2 and max = 512.
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
        '''
        Constructs the bipartite graph of the LP.
        ------------------------------------------------------------------------------------------------------------------

        Parameters:
            c : torch.tensor
                objective function coefficients

            A : torch.tensor
                constraint matrix

            b : torch.tensor
                right-hand side values for the constraints

            constraints : torch.tensor
                constraint types (0 for <= and 1 for =)

            l : torch.tensor
                lower bounds for the variables
            
            u : torch.tensor
                upper bounds for the variables
        ------------------------------------------------------------------------------------------------------------------
        '''

        # Constraint features
        hv = torch.cat((b.unsqueeze(2), constraints.unsqueeze(2)), dim=2)

        # Variable features
        hw = torch.cat((c.unsqueeze(2), l.unsqueeze(2), u.unsqueeze(2)), dim=2)

        # Edges
        E = A

        return hv, hw, E

    def init_features(self, hv, hw):
        '''
        Initializes the features of the nodes (layer 0).
        ------------------------------------------------------------------------------------------------------------------

        Parameters:
            hv : torch.tensor
                constraint features
            
            hw : torch.tensor
                variable features
        
        Returns:
            The initialized features of the nodes.
        ------------------------------------------------------------------------------------------------------------------
        '''

        # Applies MLP to each line of constraint features
        hv_0 = []
        for i in range(self.num_constraints):
            hv_0.append(self.fv_in(hv[:, i]))

        # Applies MLP to each line of variable features
        hw_0 = []
        for j in range(self.num_variables):
            hw_0.append(self.fw_in(hw[:, j]))

        hv = torch.stack(hv_0, dim=1)
        hw = torch.stack(hw_0, dim=1)

        return hv, hw

    def layer_left(self, hv, hw, E, layer):
        '''
        Update left nodes' features with MLP (constraints).
        ------------------------------------------------------------------------------------------------------------------

        Parameters:
            hv : torch.tensor
                constraint features

            hw : torch.tensor
                variable features

            E : torch.tensor
                edges

            layer : int
                layer index

        Returns:
            The updated features of left nodes.
        ------------------------------------------------------------------------------------------------------------------
        '''

        # For each left node, computes the weighted sum of the MLP output of the features of its neighbors.
        # The aggregation operator is the sum, and the update function is another MLP.
        hv_l = []
        for i in range(self.num_constraints):
            
            s = []
            for j in range(self.num_variables):
                # Multiply the edge weight connecting the left node i and the right node j by the MLP output of the right node j
                s.append(torch.mul(E[:, i, j, None], self.fw[layer](hw[:, j])))

            # Aggregation
            s = torch.sum(torch.stack(s, dim=1), dim=1)

            joint = torch.cat((hv[:, i], s), dim=1)

            # Applies second MLP to the left node i
            hv_l.append(self.gv[layer](joint))
        
        hv_l = torch.stack(hv_l, dim=1)

        return hv_l
    
    def layer_right(self, hv, hw, E, layer):
        '''
        Update right nodes' features with MLP (variables).
        ------------------------------------------------------------------------------------------------------------------

        Parameters:
            hv : torch.tensor
                constraint features

            hw : torch.tensor
                variable features

            E : torch.tensor
                edges

            layer : int
                layer index

        Returns:
            The updated features of right nodes.
        ------------------------------------------------------------------------------------------------------------------
        '''

        # For each right node, computes the weighted sum of the MLP output of the features of its neighbors.
        # The aggregation operator is the sum, and the update function is another MLP.
        hw_l = []
        for j in range(self.num_variables):

            s = []
            for i in range(self.num_constraints):
                # Multiply the edge weight connecting the left node i and the right node j by the MLP output of the left node i
                s.append(torch.mul(E[:, i, j, None], self.fv[layer](hv[:, i])))
            
            # Aggregation
            s = torch.sum(torch.stack(s, dim=1), dim=1)

            joint = torch.cat((hw[:, j], s), dim=1)
            
            # Applies second MLP to the right node j
            hw_l.append(self.gw[layer](joint))
        
        hw_l = torch.stack(hw_l, dim=1)

        return hw_l
    
    def single_output(self, hv, hw):
        '''
        Output for feasibility and objective functions.
        ------------------------------------------------------------------------------------------------------------------

        Parameters:
            hv : torch.tensor
                constraint features

            hw : torch.tensor
                variable features

        Returns:
            If the output is for feasibility, returns a binary vector indicating if the LP's are feasible or not.
            If the output is for objective, returns the objective function value of the LP's.
        '''

        y_out = self.f_out(torch.cat((torch.sum(hv, 1), torch.sum(hw, 1)), dim=1))

        return y_out
    
    def sol_output(self, hv, hw):
        '''
        Output for solution function.
        ------------------------------------------------------------------------------------------------------------------

        Parameters:
            hv : torch.tensor
                constraint features

            hw : torch.tensor
                variable features

        Returns:
            Returns the approximated solution of the LP's.
        '''

        sol = []
        for j in range(self.num_variables):
            joint = torch.cat((torch.sum(hv, 1), torch.sum(hw, 1), hw[:, j]), dim=1)
            sol.append(self.fw_out(joint))

        sol = torch.stack(sol, dim=1)

        return sol[:, :, 0]

    def forward(self, c, A, b, constraints, l, u, phi = 'feas'):
        '''
        Forward pass of the model.
        ------------------------------------------------------------------------------------------------------------------

        Parameters:
            c : torch.tensor
                objective function coefficients

            A : torch.tensor
                constraint matrix

            b : torch.tensor
                right-hand side values for the constraints

            constraints : torch.tensor
                constraint types (0 for <= and 1 for =)

            l : torch.tensor
                lower bounds for the variables
            
            u : torch.tensor
                upper bounds for the variables

            phi : str
                type of function (feas, obj or sol)

        Returns:
            If the output is for feasibility, returns a binary vector indicating if the LP's are feasible or not.
            If the output is for objective, returns the objective function value of the LP's.
            If the output is for solution, returns the approximated solution of the LP's.
        '''

        hv, hw, E = self.construct_graph(c, A, b, constraints, l, u)
        hv, hw = self.init_features(hv, hw)

        # Iterates over the layers
        for l in range(self.num_layers-1):
            old_hv = hv
            hv = self.layer_left(hv, hw, E, l)
            hw = self.layer_right(old_hv, hw, E, l)

        if phi == 'feas':
            output = self.single_output(hv,hw)
            bins = [1 if elem >= 1/2 else 0 for elem in output]
            return torch.tensor(bins, dtype=torch.float32, requires_grad=True)
        
        elif phi == 'obj':
            return self.single_output(hv,hw)
        
        elif phi == 'sol':
            return self.sol_output(hv,hw)
        
        else:
            return "Please, choose one type of function: feas, obj or sol"
