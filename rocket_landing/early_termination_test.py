import numpy as np
import os
import cvxpy as cp
from cvxpygen import cpg
import scipy
from utils import socp_export_data_to_c, replace_in_file


def stage_cost_expansion(p, k):
    dx = -np.array(p['Xref'][k])
    du = -np.array(p['Uref'][k])
    Q = p['Q']
    R = p['R']
    return Q, Q @ dx, R, R @ du


def term_cost_expansion(p):
    dx = -np.array(p['Xref'][p['NHORIZON']-1])
    Qf = p['Qf']
    return Qf, Qf @  dx


def update_linear_term(params):
    # Define cost
    P = np.zeros((NN, NN))
    q = np.zeros((NN, 1))
    for j in range(NHORIZON - 1):
        Q, dQ, R, dR = stage_cost_expansion(params, j)
        Pj = np.block([[Q, np.zeros((NSTATES, NINPUTS))],
                      [np.zeros((NINPUTS, NSTATES)), R]])
        qj = np.concatenate((dQ, dR))
        # import pdb; pdb.set_trace()
        P[j * (NSTATES + NINPUTS): (j + 1) * (NSTATES + NINPUTS), j * (NSTATES + NINPUTS): (j + 1) * (NSTATES + NINPUTS)] = Pj
        q[j * (NSTATES + NINPUTS): (j + 1) * (NSTATES + NINPUTS)] = qj.reshape(-1, 1)
    
    # Terminal cost
    Qf, dQf = term_cost_expansion(params)
    P[-NSTATES:, -NSTATES:] = Qf  
    q[-NSTATES:] = dQf.reshape(-1, 1)    
    return P, q


# Define problem parameters
Ad = np.array([[1.0, 0.0, 0.0, 0.05, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.05, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.05],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
Bd = np.array([[0.000125, 0.0, 0.0],
                [0.0, 0.000125, 0.0],
                [0.0, 0.0, 0.000125],
                [0.005, 0.0, 0.0],
                [0.0, 0.005, 0.0],
                [0.0, 0.0, 0.005]])
fd = np.array([0.0, 0.0, -0.0122625, 0.0, 0.0, -0.4905])

NSTATES = 6
NINPUTS = 3
NHORIZON = 32 # horizon length, short for MPC
NTOTAL = 301
dt = 0.05
t_vec = dt * np.arange(NTOTAL)
# print('t_vec', t_vec.shape)

x0 = np.array([4, 2, 20, -3, 2, -4.5])
xg = np.array([0, 0, 0, 0, 0, 0.0])
Xref = [x0 + (xg - x0) * k / (NTOTAL-1) for k in range(NTOTAL)]
Xref = Xref + [xg for _ in range(NHORIZON)]
Uref = [[0, 0, 10.] for _ in range(NTOTAL - 1)]
# import pdb; pdb.set_trace()
Xref_hrz = Xref[:NHORIZON]*1
Uref_hrz = Uref[:NHORIZON-1]*1

# print('Xref', Xref)
# print('Uref', len(Uref))
Q = 1e3 * np.eye(NSTATES)
R = 1e0 * np.eye(NINPUTS)
Qf = Q*1

# gravity = np.array([0, 0, -9.81])
# mass = 10.0
# perWeightMax = 2.0
# θ_thrust_max = 5.0  # deg

# Sloppy bounds to test
u_min = -10 * np.ones(NINPUTS)
u_max = 105.0 * np.ones(NINPUTS)
x_min = [-5, -5, 0, -10, -10, -10.0]
x_max = [5, 5, 20, 10, 10, 10.0]

params = {
    'NSTATES': NSTATES,
    'NINPUTS': NINPUTS,
    'NHORIZON': NHORIZON,
    'Q': Q,
    'R': R,
    'Qf': Qf,
    'u_min': u_min,
    'u_max': u_max,
    'x_min': x_min,
    'x_max': x_max,
    'Xref': Xref_hrz,
    'Uref': Uref_hrz,
    'dt': dt,
}

X = [np.copy(x0) for _ in range(NHORIZON)]
# print('X', len(X))
U = [np.copy(Uref[0]) for _ in range(NHORIZON-1)]
# print('U', len(U))

NN = NHORIZON*NSTATES + (NHORIZON-1)*NINPUTS  # Number of decision variables
x0_param = cp.Parameter(NSTATES)  # initial state

z = cp.Variable(NN)
inds = np.reshape(np.arange(0, NN + NINPUTS), (NSTATES + NINPUTS, NHORIZON), order='F')
# print('inds', inds.shape)
xinds = [inds[:NSTATES, i] for i in range(NHORIZON)]
# print('xinds', (xinds))
uinds = [inds[NSTATES:, i] for i in range(NHORIZON - 1)]
# print('uinds', (uinds))

# import pdb; pdb.set_trace()

q_param = cp.Parameter((NN, 1), value =np.zeros((NN, 1)))
P, q_param.value = update_linear_term(params)  # P is unchanged
Psqrt = scipy.linalg.sqrtm(P)

objective = cp.Minimize(0.5 * cp.sum_squares(Psqrt @ z) + q_param.T @ z)
#objective = cp.Minimize(0.5 * cp.quad_form(z, P_param) + q_param.T @ z)
constraints = []

# Dynamics Constraints
for k in range(NHORIZON-1):
    constraints.append(Ad @ z[xinds[k]] + Bd @ z[uinds[k]] + fd == z[xinds[k+1]])

# Initial states
constraints.append(z[xinds[0]] == x0_param)

# Thrust angle constraint (SOC): norm([u1,u2]) <= alpha_max * u3
for k in range(NHORIZON-1):
    u1, u0 = z[uinds[k]][0:2], z[uinds[k]][2]
    constraints.append(cp.norm(u1) <= 0.25 * u0)

# Input constraints
for k in range(NHORIZON-1):
    constraints.append(z[uinds[k]] <= params['u_max'])
    constraints.append(z[uinds[k]] >= params['u_min'])

problem = cp.Problem(objective,constraints)

# MPC loop
np.random.seed(1)
Xhist = np.zeros((NSTATES, NTOTAL))
Xhist[:, 0] = x0*1.1
Uhist = np.zeros((NINPUTS, NTOTAL-1))
x0_param.value = Xhist[:, 0]*1
params["Xref"] = Xref[0:NHORIZON]*1
params["Uref"] = Uref[0:NHORIZON-1]*1
q_param.value = update_linear_term(params)[1]  

# GENERATE CODE (uncomment to generate code)
GEN_CODE = 1

opts = {"verbose": False, "max_iters": 10, "abs_tol": 1e-6, "rel_tol": 1e-6}
SOLVER = "ECOS"
# SOLVER = "SCS"

for k in range(1):
    # Get measurements
    x0_param.value = Xhist[:, k]*1

    # Update references
    params["Xref"] = Xref[k:k+NHORIZON]*1
    # params["Uref"] = Uref[k:k+NHORIZON-1]*1
    q_param.value = update_linear_term(params)[1]
    # print(q_param.value, x0_param.value)
    
    # Solve MPC problem
    if SOLVER == "ECOS":
        problem.solve(verbose=True, solver=SOLVER, maxit=opts["max_iters"], abstol=opts["abs_tol"], reltol=opts["rel_tol"])
    if SOLVER == "SCS":
        problem.solve(verbose=True, solver=SOLVER, max_iters=opts["max_iters"], eps=opts["abs_tol"], acceleration_lookback=0)
    # print(z.value)
    # Extract results
    for j in range(NHORIZON-1):
        X[j] = z[xinds[j]].value
        U[j] = z[uinds[j]].value
    X[NHORIZON-1] = z[xinds[NHORIZON-1]].value
    
    # CHECK DYNAMICS CONSTRAINT SATISFACTION


    Uhist[:, k] = U[0]*1
    print(Uhist[:, k])

    # Simulate system
    Xhist[:, k+1] = Ad @ Xhist[:, k] + Bd @ Uhist[:, k] + fd
    print(Xhist[:, k+1])
    # import pdb; pdb.set_trace()

# print(U)
# print(X)
# ## plot results
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(Xhist[:3,:NTOTAL-1].T)
# # plt.plot(np.array(X))
# plt.title('States')
# plt.show()

# plt.figure()
# plt.plot(Uhist[:,:NTOTAL-1].T)
# # plt.plot(np.array(X))
# plt.title('Controls')
# plt.show()

