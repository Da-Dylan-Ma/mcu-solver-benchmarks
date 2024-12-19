import numpy as np
import scipy.sparse as sparse

def save_smp_yaml(filename, qp_data):
    """Save QP data in SMP YAML format."""
    with open(filename, "w") as f:
        f.write('"Description": |\n')
        f.write("  MPC QP problem in SMP format\n\n")

        f.write('"Fixed": |\n')
        f.write("  0\n\n")

        f.write('"Inequality": |\n')
        buf = []
        buf.append("%%MatrixMarket matrix coordinate real general")
        rows, cols = qp_data["Aineq"].shape
        coo = qp_data["Aineq"].tocoo()
        buf.append(f"{rows} {cols} {coo.nnz}")
        for r, c, v in zip(coo.row, coo.col, coo.data):
            buf.append(f"{r + 1} {c + 1} {v:.12g}")
        for line in buf:
            f.write("  " + line + "\n")
        f.write("\n")

        f.write('"Inequality l-bounds": |\n')
        buf = []
        buf.append("%%MatrixMarket matrix array real general")
        m = len(qp_data["lineq"])
        buf.append(f"{m} 1")
        for val in qp_data["lineq"]:
            buf.append(f"{val:.12g}")
        for line in buf:
            f.write("  " + line + "\n")
        f.write("\n")

        f.write('"Inequality u-bounds": |\n')
        buf = []
        buf.append("%%MatrixMarket matrix array real general")
        m = len(qp_data["uineq"])
        buf.append(f"{m} 1")
        for val in qp_data["uineq"]:
            buf.append(f"{val:.12g}")
        for line in buf:
            f.write("  " + line + "\n")
        f.write("\n")

        f.write('"Linear": |\n')
        buf = []
        buf.append("%%MatrixMarket matrix array real general")
        m = len(qp_data["q"])
        buf.append(f"{m} 1")
        for val in qp_data["q"]:
            buf.append(f"{val:.12g}")
        for line in buf:
            f.write("  " + line + "\n")
        f.write("\n")

        f.write('"Quadratic": |\n')
        buf = []
        buf.append("%%MatrixMarket matrix coordinate real general")
        rows, cols = qp_data["H"].shape
        coo = qp_data["H"].tocoo()
        buf.append(f"{rows} {cols} {coo.nnz}")
        for r, c, v in zip(coo.row, coo.col, coo.data):
            buf.append(f"{r + 1} {c + 1} {v:.12g}")
        for line in buf:
            f.write("  " + line + "\n")
        f.write("\n")

        f.write('"Equality": |\n')
        buf = []
        buf.append("%%MatrixMarket matrix coordinate real general")
        rows, cols = qp_data["Aeq"].shape
        coo = qp_data["Aeq"].tocoo()
        buf.append(f"{rows} {cols} {coo.nnz}")
        for r, c, v in zip(coo.row, coo.col, coo.data):
            buf.append(f"{r + 1} {c + 1} {v:.12g}")
        for line in buf:
            f.write("  " + line + "\n")
        f.write("\n")

        f.write('"Equality bounds": |\n')
        buf = []
        buf.append("%%MatrixMarket matrix array real general")
        m = len(qp_data["beq"])
        buf.append(f"{m} 1")
        for val in qp_data["beq"]:
            buf.append(f"{val:.12g}")
        for line in buf:
            f.write("  " + line + "\n")
        f.write("\n")


data = np.load('rand_prob_osqp_params.npz')

nx = int(data['nx'])
nu = int(data['nu'])
Nh = int(data['Nh'])
Q = data['Q'] * 1000
R = data['R']
QN = data['Qf']
A = data['A']
B = data['B']
x0 = np.zeros(nx)
xbar_full = data['x_bar'].T
umin = data['umin'][:, 0]
umax = data['umax'][:, 0]
xmin = -np.inf*np.ones(nx)
xmax = np.inf*np.ones(nx)

H = sparse.block_diag([sparse.kron(sparse.eye(Nh), Q), QN,
                       sparse.kron(sparse.eye(Nh), R)], format='csc')
q = np.hstack([np.hstack([-Q @ xbar_full[:, i] for i in range(Nh)]), -QN @ xbar_full[:, Nh], np.zeros(Nh * nu)])

Ax = sparse.kron(sparse.eye(Nh + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(Nh + 1, k=-1), sparse.csc_matrix(A))
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, Nh)), sparse.eye(Nh)]), sparse.csc_matrix(B))
Aeq = sparse.hstack([Ax, Bu])
beq = np.hstack([-x0, np.zeros(Nh * nx)])

Aineq = sparse.eye((Nh + 1) * nx + Nh * nu)
lineq = np.hstack([np.kron(np.ones(Nh + 1), xmin), np.kron(np.ones(Nh), umin)])
uineq = np.hstack([np.kron(np.ones(Nh + 1), xmax), np.kron(np.ones(Nh), umax)])

qp_data = {
    "H": H,
    "q": q,
    "Aeq": Aeq,
    "beq": beq,
    "Aineq": Aineq,
    "lineq": lineq,
    "uineq": uineq,
}

save_smp_yaml("qp_problem.yml", qp_data)
