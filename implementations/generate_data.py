import numpy as np
import torch
import random as rd
from scipy.optimize import linprog

def generate_random_linear_program(num_variables, num_constraints, nnz = 5):
    # Generate random coefficients for the objective function
    c = np.random.uniform(-1, 1, num_variables) * 0.01

    # Generate random coefficients for the constraint matrix
    A = np.zeros((num_constraints, num_variables))
    EdgeIndex = np.zeros((nnz, 2))
    EdgeIndex1D = rd.sample(range(num_constraints * num_variables), nnz)
    EdgeFeature = np.random.normal(0, 1, nnz)
        
    for l in range(nnz):
        i = int(EdgeIndex1D[l] / num_variables)
        j = EdgeIndex1D[l] - i * num_variables
        EdgeIndex[l, 0] = i
        EdgeIndex[l, 1] = j
        A[i, j] = EdgeFeature[l]

    # Generate random right-hand side values for the constraints
    b = np.random.uniform(-1, 1, num_constraints)

    bounds = np.random.normal(0, 10, size = (num_variables, 2))
        
    for j in range(num_variables):
        if bounds[j, 0] > bounds[j, 1]:
            temp = bounds[j, 0]
            bounds[j, 0] = bounds[j, 1]
            bounds[j, 1] = temp

    # Generate random constraint types (0 for <= and 1 for =)
    constraint_types = np.random.choice([0, 1], size=num_constraints, p=[0.7, 0.3])

    # Adjust the constraint matrix and right-hand side based on the constraint types
    for i in range(num_constraints):
        if constraint_types[i] == 1:  # Equality constraint
            b[i] = np.dot(A[i], np.random.rand(num_variables))  # Randomly generate a feasible solution

    # Return the generated linear program
    return c, A, b, constraint_types, bounds


def solve_linear_program(c, A, b, constraint_types, bounds):
    # Solve the linear program
    res = linprog(c, A_ub=A[constraint_types == 0], b_ub=b[constraint_types == 0],
                  A_eq=A[constraint_types == 1], b_eq=b[constraint_types == 1], bounds=bounds)

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
                                                                               num_constraints)

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
                                                                               num_constraints)
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
            batches_solutions.append(torch.tensor(solution, dtype=torch.float32))
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