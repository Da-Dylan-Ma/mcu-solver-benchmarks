"""
File: prob_data_gen.py
Author: Anoushka, Khai
Date: 2023-09-15
Description: A Python script to generate random MPC problem for OSQP (Python) and TinyMPC (C++).
"""

import numpy as np
import os
import scipy.sparse as sp

def export_xref_to_c(declare, data, nx):
    string = declare + "= {\n"
    for i in range(data.shape[0]):
        string = string + "  "
        for j in range(nx):
            if i == data.shape[0] and j == nx:
                this_str = str(data[i][j]) + ",\t"
            else:
                this_str = str(data[i][j]) + ",\t"
            # str = str * this_str * "f"
            string = string + this_str
        string = string + "\n"
    string = string + "};"
    return string

def export_mat_to_c(declare, data):
    string = declare + "= {\n"
    for i in range(data.shape[0]):
        string = string + "  "
        for j in range(data.shape[1]):
            if i == data.shape[0] and j == data.shape[1]:
                this_str = str(data[i][j]) + ",\t"
            else:
                 this_str = str(data[i][j]) + ",\t"
            string = string + this_str
        string = string + "\n"
    string = string + "};"
    return string

def tinympc_export_data_to_c(xbar, A, B, Q, Qf, R, umin, umax, xmin, xmax, nx, nu, Nh, Nsim):
    include_statement = '#include "types.hpp"\n\n'
    boilerplate = "#pragma once\n\n"

    xbar_string = export_xref_to_c("const PROGMEM tinytype Xref_data["+str(Nsim)+"*"+str(nx)+"] ", xbar, nx)

    A_data_string = export_mat_to_c("const PROGMEM tinytype Adyn_data["+str(nx)+"*"+str(nx)+"] ", A) + "\n\n"
    B_data_string = export_mat_to_c("const PROGMEM tinytype Bdyn_data["+str(nx)+"*"+str(nu)+"] ", B) + "\n\n"
   
    Q_diag = np.diagonal(Q)
    Q_data_string = "const PROGMEM tinytype Q_data["+str(nx)+"] = {"
    for i in range(nx):
        Q_data_string = Q_data_string + str(Q_diag[i])
        if i != nx-1:
            Q_data_string = Q_data_string + ','
    Q_data_string = Q_data_string+"};\n\n"
    
    Qf_diag = np.diagonal(Qf)
    Qf_data_string = "const PROGMEM tinytype Qf_data["+str(nx)+"] = {"
    for i in range(nx):
        Qf_data_string = Qf_data_string + str(Qf_diag[i])
        if i != nx-1:
            Qf_data_string = Qf_data_string + ','
    Qf_data_string = Qf_data_string+"};\n\n"
    
    R_diag = np.diagonal(R)
    R_data_string = "const PROGMEM tinytype R_data["+str(nu)+"] = {"
    for i in range(nu):
        R_data_string = R_data_string + str(R_diag[i])
        if i != nu-1:
            R_data_string = R_data_string + ','
    R_data_string = R_data_string+"};\n\n"

    umin_string = export_mat_to_c("const PROGMEM tinytype umin["+str(nu)+"] ", umin) + "\n\n"
    umax_string = export_mat_to_c("const PROGMEM tinytype umax["+str(nu)+"] ", umax) + "\n\n"
    xmin_string = export_mat_to_c("const PROGMEM tinytype xmin["+str(nx)+"] ", xmin) + "\n\n"
    xmax_string = export_mat_to_c("const PROGMEM tinytype xmax["+str(nx)+"] ", xmax) + "\n\n"

    f = open('random_problems/prob_nx_'+str(nx)+'/constants.hpp','w')
    f.write("#define NSTATES "+str(nx)+'\n\n')
    f.write("#define NINPUTS "+str(nu)+'\n\n')
    f.write("#define NHORIZON "+str(Nh)+'\n\n')
    f.write("#define NTOTAL "+str(Nsim)+'\n\n')
    f.write('#define NSTATE_CONSTRAINTS 1\n\n')

    f.close()

    f = open('random_problems/prob_nx_'+str(nx)+"/rand_prob_tinympc_xbar.hpp", "w")
    # f = open("rand_prob_tinympc_xbar.hpp", "w")
    f.write(include_statement)
    f.write(boilerplate)
    f.write(xbar_string)
    f.close()

    f = open('random_problems/prob_nx_'+str(nx)+"/rand_prob_tinympc_params.hpp", "w")
    # f = open("rand_prob_tinympc_params.hpp", "w")
    f.write(include_statement)
    f.write(boilerplate)
    f.write(A_data_string)
    f.write(B_data_string)
    f.write(Q_data_string)
    f.write(Qf_data_string)
    f.write(R_data_string)
    f.write(umin_string)
    f.write(umax_string)
    f.write(xmin_string)
    f.write(xmax_string)
        
    K = np.zeros((nu,nx))
    P = np.zeros((nx,nx))
    Kprev = np.zeros((nu,nx))
    Pprev = np.zeros((nx,nx))

    # Compute Kinf, Pinf
    riccati_iters = 0
    riccati_err = 1e-10
    Pprev = Qf
    while True:
        K = np.linalg.inv(R + B.T @ Pprev @ B) @ (B.T @ Pprev @ A)
        P = Q + A.T @ Pprev @ (A - B@K)
        if np.max(np.abs(K - Kprev)) < riccati_err:
            break
        Kprev = K
        Pprev = P
        riccati_iters += 1

    # Cache precomputed values
    rho = 0.1
    Q = Q + rho*np.eye(nx)
    Qf = P + rho*np.eye(nx)
    R = R + rho*np.eye(nu)

    # Compute Kinf, Pinf
    riccati_iters = 0
    riccati_err = 1e-10
    Pprev = Qf
    while True:
        K = np.linalg.inv(R + B.T @ Pprev @ B) @ (B.T @ Pprev @ A)
        P = Q + A.T @ Pprev @ (A - B@K)
        if np.max(np.abs(K - Kprev)) < riccati_err:
            break
        Kprev = K
        Pprev = P
        riccati_iters += 1

    Kinf = K
    Pinf = P
    Quu_inv = np.linalg.inv(R + B.T @ Pinf@B)
    AmBKt = (A - B@K).T
    coeff_d2p_list = Kinf.T@R - AmBKt@Pinf@B

    rho_string = "const PROGMEM tinytype rho_value = "+str(rho)+";\n\n"
    Kinf_string = export_mat_to_c("const PROGMEM tinytype Kinf_data["+str(nu)+"*"+str(nx)+"] ", Kinf) + "\n\n"
    Pinf_string = export_mat_to_c("const PROGMEM tinytype Pinf_data["+str(nx)+"*"+str(nx)+"] ", Pinf) + "\n\n"
    Quu_inv_string = export_mat_to_c("const PROGMEM tinytype Quu_inv_data["+str(nu)+"*"+str(nu)+"] ", Quu_inv) + "\n\n"
    AmBKt_string = export_mat_to_c("const PROGMEM tinytype AmBKt_data["+str(nx)+"*"+str(nx)+"] ", AmBKt) + "\n\n"
    coeff_d2p_list_string = export_mat_to_c("const PROGMEM tinytype coeff_d2p_data["+str(nx)+"*"+str(nu)+"] ", coeff_d2p_list) + "\n\n"

    f.write(rho_string)
    f.write(Kinf_string)
    f.write(Pinf_string)
    f.write(Quu_inv_string)
    f.write(AmBKt_string)
    f.write(coeff_d2p_list_string)
    f.close()

def osqp_export_data_to_python(xbar, A, B, Q, Qf, R, umin, umax, xmin, xmax, nx, nu, Nh, Nsim):
    np.savez('random_problems/prob_nx_'+str(nx)+'/rand_prob_osqp_params.npz',nx=nx, nu=nu, Nh=Nh, Nsim=Nsim, Q=Q, R=R, A=A, B=B, Qf=Qf, x_bar=xbar, umin=umin, umax=umax, xmin=xmin, xmax=xmax)
    # np.savez('rand_prob_osqp_params.npz',nx=nx, nu=nu, Nh=Nh, Nsim=Nsim, Q=Q, R=R, A=A, B=B, Qf=Qf, x_bar=xbar, umin=umin, umax=umax, xmin=xmin, xmax=xmax)
    SIZE_Q = (Nh + 1) * nx + Nh * nu
    SIZE_LU = (Nh + 1) * nx * 2 + Nh * nu

    include_statement = '#include "osqp_api_types.h"\n\n'
    boilerplate = "#pragma once\n\n"

    xbar_string = export_xref_to_c("const PROGMEM OSQPFloat Xref_data["+str(Nsim)+"*"+str(nx)+"] ", xbar, nx)+'\n\n'

    # f = open("rand_prob_osqp_xbar.h", "w")
    f = open('random_problems/prob_nx_'+str(nx)+"/rand_prob_osqp_xbar.h", "w")
    f.write(include_statement)
    f.write(boilerplate)
    f.write(xbar_string)

    f.write("#define NSTATES "+str(nx)+'\n\n')
    f.write("#define NINPUTS "+str(nu)+'\n\n')
    f.write("#define NHORIZON "+str(Nh)+'\n\n')
    f.write("#define NTOTAL "+str(Nsim)+'\n\n')
    f.write("#define SIZE_Q "+str(SIZE_Q)+'\n\n')
    f.write("#define SIZE_LU "+str(SIZE_LU)+'\n\n')

    Q_string = export_mat_to_c("const PROGMEM OSQPFloat mQ["+str(nx)+"*"+str(nx)+"] ", -Q)+'\n\n'
    Qf_string = export_mat_to_c("const PROGMEM OSQPFloat mQf["+str(nx)+"*"+str(nx)+"] ", -Qf)+'\n\n'
    A_string = export_mat_to_c("const PROGMEM OSQPFloat A["+str(nx)+"*"+str(nx)+"] ", A)+'\n\n'
    B_string = export_mat_to_c("const PROGMEM OSQPFloat B["+str(nx)+"*"+str(nu)+"] ", B)+'\n\n'

    f.write(Q_string)
    f.write(Qf_string)
    f.write(A_string)
    f.write(B_string)

    f.close()

def nasoq_export_data_to_c(xbar, A, B, Q, Qf, R, umin, umax, xmin, xmax, nx, nu, Nh, Nsim):
    # Dimensions of the stacked problem
    # States: (Nh+1)*nx
    # Inputs: Nh*nu
    Nstates = (Nh + 1)*nx
    Ninputs = Nh*nu
    Nz = Nstates + Ninputs

    # Construct Hessian H:
    # H is block diagonal:
    # [Q, Q, ..., Q, Qf, R, R, ..., R]
    Q_blocks = [Q for _ in range(Nh)] + [Qf]
    R_blocks = [R for _ in range(Nh)]
    H_blocks = Q_blocks + R_blocks
    H = sp.block_diag(H_blocks, format='csc')

    # Construct reference z_ref:
    # xbar is Nsim-by-nx of reference states. (TODO: need a double-check)
    # Use the first (Nh+1) entries of xbar for state reference along the horizon:
    if Nsim < (Nh+1):
        raise ValueError("Not enough xbar steps for the given horizon.")
    x_ref = xbar[:(Nh+1), :]

    u_ref = np.zeros((Nh, nu))
    z_ref = np.concatenate([x_ref.flatten(), u_ref.flatten()])

    # Compute q = -H z_ref
    q = -H.dot(z_ref)

    # Construct equality constraints:
    # For k=0,...,Nh-1:
    # x_{k+1} - A x_k - B u_k = 0
    # And initial condition: x_0 = x_ref(0)
    # Aeq z = beq:
    # Variables order: z = [x_0; x_1; ...; x_Nh; u_0; ...; u_{Nh-1}]
    # Size: Nstates = (Nh+1)*nx, Ninputs = Nh*nu

    Aeq_init = sp.hstack([sp.eye(nx), sp.csc_matrix((nx, Nstates - nx + Ninputs))], format='csc')
    beq_init = x_ref[0]

    # Dynamics constraints:
    # For k=0,...,Nh-1:
    # x_{k+1} - A x_k - B u_k = 0
    # Row structure: [0..x_k..x_{k+1}..u_k..]
    Aeq_dyn_blocks = []
    beq_dyn = []
    for k in range(Nh):
        # x_{k+1} index: (k+1)*nx to (k+2)*nx - 1
        # x_k index: k*nx to (k+1)*nx - 1
        # u_k index: Nstates + k*nu to Nstates + (k+1)*nu -1
        row_x_next = sp.lil_matrix((nx, Nz))
        # x_{k+1}
        row_x_next[:, (k+1)*nx:(k+2)*nx] = sp.eye(nx)
        # -A x_k
        row_x_next[:, k*nx:(k+1)*nx] = -A
        # -B u_k
        row_x_next[:, Nstates + k*nu: Nstates + (k+1)*nu] = -B
        Aeq_dyn_blocks.append(row_x_next.tocsc())
        beq_dyn.append(np.zeros(nx))

    Aeq_dyn = sp.vstack(Aeq_dyn_blocks, format='csc')
    beq_dyn = np.concatenate(beq_dyn)

    # Full equality matrix and vector:
    Aeq = sp.vstack([Aeq_init, Aeq_dyn], format='csc')
    beq = np.concatenate([beq_init, beq_dyn])

    # Convert equalities to inequalities:
    # Aeq z = beq  => Aeq z ≤ beq and -Aeq z ≤ -beq
    Aeq_all = sp.vstack([Aeq, -Aeq], format='csc')
    beq_all = np.concatenate([beq, -beq])

    # Inequality constraints from state and input bounds:
    # For each x_k: xmin ≤ x_k ≤ xmax
    #    => x_k ≤ xmax and -x_k ≤ -xmin
    # For each u_k: umin ≤ u_k ≤ umax
    #    => u_k ≤ umax and -u_k ≤ -umin
    #
    # Construct block matrices for these:
    # State bounds: (Nh+1)*nx states
    # x ≤ xmax => (I for states, 0 for inputs)
    I_states = sp.eye(Nstates, format='csc')
    I_inputs = sp.eye(Ninputs, format='csc')
    zero_si = sp.csc_matrix((Ninputs, Nstates))  # for u-bounds
    zero_is = sp.csc_matrix((Nstates, Ninputs))  # for x-bounds

    # x ≤ xmax
    C_xmax = sp.hstack([I_states, zero_is], format='csc')
    d_xmax = xmax.flatten().repeat(Nh+1)

    # -x ≤ -xmin => multiply by -1:
    C_xmin = sp.hstack([-I_states, zero_is], format='csc')
    d_xmin = -xmin.flatten().repeat(Nh+1)

    # u ≤ umax
    C_umax = sp.hstack([zero_si, I_inputs], format='csc')
    d_umax = umax.flatten().repeat(Nh)

    # -u ≤ -umin
    C_umin = sp.hstack([zero_si, -I_inputs], format='csc')
    d_umin = -umin.flatten().repeat(Nh)

    # Stack all inequality constraints:
    C_ineq = sp.vstack([Aeq_all, C_xmax, C_xmin, C_umax, C_umin], format='csc')
    d_ineq = np.concatenate([beq_all, d_xmax, d_xmin, d_umax, d_umin])

    # Extract lower-triangular part of H:
    H_lower = sp.tril(H).tocsc()
    Hp, Hi, Hx = H_lower.indptr, H_lower.indices, H_lower.data
    Cp, Ci, Cx = C_ineq.indptr, C_ineq.indices, C_ineq.data

    # Create output directory
    output_dir = 'random_problems/prob_nx_'+str(nx)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    hpp_file = os.path.join(output_dir, "rand_prob_nasoq_data.hpp")

    with open(hpp_file, "w") as f:
        f.write("#pragma once\n\n")
        f.write('#include "osqp_api_types.h"\n')
        f.write('#include <stddef.h>\n\n')

        # Dimensions
        f.write("#define NSTATES "+str(nx)+"\n")
        f.write("#define NINPUTS "+str(nu)+"\n")
        f.write("#define NHORIZON "+str(Nh)+"\n")
        f.write("#define NTOTAL "+str(Nsim)+"\n")
        f.write("// Number of decision variables\n")
        f.write("#define NZ "+str(Nz)+"\n\n")

        def write_array(name, arr, dtype='OSQPFloat'):
            f.write("const "+dtype+" "+name+"["+str(len(arr))+"] = {")
            f.write(",".join(map(str, arr)))
            f.write("};\n\n")

        def write_int_array(name, arr):
            f.write("const OSQPInt "+name+"["+str(len(arr))+"] = {")
            f.write(",".join(map(str, arr)))
            f.write("};\n\n")

        # Write H, C, q, d
        write_int_array("Hp", Hp)
        write_int_array("Hi", Hi)
        write_array("Hx", Hx, 'OSQPFloat')

        write_int_array("Cp", Cp)
        write_int_array("Ci", Ci)
        write_array("Cx", Cx, 'OSQPFloat')

        write_array("q", q, 'OSQPFloat')
        write_array("d", d_ineq, 'OSQPFloat')

def generate_data(nx, nu, Nh, Nsim):
    np.random.seed(123)
    # Generate Q: Q_{ii} = U(0, 1)
    Q_diag = np.random.uniform(0,10,nx)
    Q = np.diag(Q_diag)

    # Generate R: R_{ii} = 0.1
    R_diag = 0.1*np.ones(nu)
    R = np.diag(R_diag)

    # Generate Qf: (N-1)*Q
    Qf = ((Nh)-1)*Q

    # reference trajectory xk, uk --> uk ~ N(0, 1), xk+1 = f(xk, uk)
    u = np.random.normal(size=(nu, Nsim-1))

    # A and B are random matrices with eigenvalues inside of the unit circle

    # Generate a controllable system
    while True:
        # SVD for both A and B to ensure that the eigenvalues are inside of the unit circle
        A = np.random.uniform(low=-1, high=1, size=(nx, nx))
        U, S, Vh = np.linalg.svd(A) # S is a vector of the non-zero singular values
        E = np.zeros((U.shape[0], Vh.shape[0])) # E is the SVD matrix of singular values Σ 
        S = S / np.max(S) # Scale the singular values so that 
        np.fill_diagonal(E, S)
        A = U @ E @ Vh.T

        B = np.random.uniform(low=-1, high=1, size=(nx,nu))

        # Check if the system is controllable
        C = np.zeros((nx,nx*nu)) # controllability matrix
        Ak = np.zeros((nx,nx)) + np.eye(nx)
        for k in range(nx):
            C[:, (k)*nu:(k)*nu+nu] = Ak@B
            Ak = Ak @ A

        if np.linalg.matrix_rank(C) == A.shape[0]: # only true if system is controllable
            break
        else:
            print("Not controllable")

    umax = np.zeros((nu,1)) + 3
    umin = np.zeros((nu,1)) - 3
    xmin = np.zeros((nx,1)) - 10000 # might add state constraints later
    xmax = np.zeros((nx,1)) + 10000

    x0 = np.zeros(nx)
    xbar = np.zeros((Nsim, nx))
    xbar[0,:] = x0

    for k in range(Nsim-1):
        xbar[k+1,:] = A @ xbar[k,:] + B @ u[:,k]


    ### SAVE ALL RANDOM PROBLEM DATA TO A HPP FILE TO BE USED BY ALL SOLVERS ###

    tinympc_export_data_to_c(xbar, A, B, Q, Qf, R, umin, umax, xmin, xmax, nx, nu, Nh, Nsim)
    osqp_export_data_to_python(xbar, A, B, Q, Qf, R, umin, umax, xmin, xmax, nx, nu, Nh, Nsim)
    nasoq_export_data_to_c(xbar, A, B, Q, Qf, R, umin, umax, xmin, xmax, nx, nu, Nh, Nsim)
        
if __name__ == '__main__':
    ## Vary Nh
    # nx = 10
    # nu = 4
    # Nh = 10
    # Nsim = 200

    # for Nh in [8, 10, 12, 14, 16, 25, 50, 75, 100]:
    #     os.system('mkdir random_problems/prob_Nh_'+str(Nh))
    #     generate_data(nx, nu, Nh, Nsim)

    ## Vary nx
    nu = 4
    Nh = 10
    Nsim = 200

    for nx in [10]:
        os.system('mkdir random_problems/prob_nx_'+str(nx))
        generate_data(nx, nu, Nh, Nsim)
        print('generated: random_problems/prob_nx_'+str(nx))

    # # ## Vary nu
    # nx = 10
    # nu = 4
    # Nh = 10
    # Nsim = 200

    # for nu in range(2,30,2):
    #     os.system('mkdir random_problems/prob_nu_'+str(nu))
    #     generate_data(nx, nu, Nh, Nsim)
