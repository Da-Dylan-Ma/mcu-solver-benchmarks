/******************************************************************************
 * Teensy Eigen‑only NASOQ QP Solver (No Pivoting, No Supernodes)
 *****************************************************************************/

#include <Arduino.h>
#undef min
#undef max
#undef abs
#undef B1
#undef F

#include <../include/eigen/Eigen/Sparse>
#include <../include/eigen/Eigen/SparseCholesky>
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

// Put everything in our nasoq namespace
namespace nasoq {

  // -------------------------------
  // Minimal CSC Matrix Structure
  // (This is how NASOQ represents the matrix in compressed‐column form.)
  struct CSC {
    int nrow;      // Number of rows
    int ncol;      // Number of columns
    int *p;        // Column pointer array of size ncol+1
    int *i;        // Row indices array (length: nzmax)
    double *x;     // Nonzero numerical values (length: nzmax)
    int nzmax;     // Maximum number of nonzeros
    int stype;     // Symmetry type (-1 for unsymmetric)
    int packed;    // Flag: 1 if packed
    int sorted;    // Flag: 1 if row indices are sorted
    int xtype;     // Type indicator for x (1 for real)
  };

  // -------------------------------
  // Utility: Convert a CSC matrix to an Eigen::SparseMatrix.
  // This step plays the role of the symbolic analysis in our Eigen version.
  Eigen::SparseMatrix<double> cscToEigen(const CSC &A) {
    int n = A.ncol;
    int m = A.nrow;
    std::vector<Eigen::Triplet<double>> triplets;
    for (int j = 0; j < n; j++) {
      for (int idx = A.p[j]; idx < A.p[j+1]; idx++) {
        int row = A.i[idx];
        double val = A.x[idx];
        triplets.push_back(Eigen::Triplet<double>(row, j, val));
      }
    }
    Eigen::SparseMatrix<double> eigenA(m, n);
    eigenA.setFromTriplets(triplets.begin(), triplets.end());
    return eigenA;
  }

  // -------------------------------
  // Eigen-only SolverSettings: This class wraps the conversion,
  // factorization, and solution steps. The original NASOQ SolverSettings
  // and its routines (symbolic_analysis, numerical_factorization, solve_only)
  // have been replaced by the following.
  class SolverSettings {
  public:
    CSC *A;                       // Input matrix (CSC format)
    double *rhs;                  // Right-hand side vector (external)
    Eigen::SparseMatrix<double> eigenA;  // Eigen representation of A
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver; // LDL^T factorization (non-pivoting)
    Eigen::VectorXd x;            // Eigen solution vector

    // Constructor: simply stores pointers to the external A and rhs.
    SolverSettings(CSC *Amat, double *rhs_in)
      : A(Amat), rhs(rhs_in) { }

    // Symbolic Analysis: convert CSC to an Eigen sparse matrix.
    // (In the original NASOQ, this would also compute elimination reach, supernodes, etc.)
    int symbolic_analysis() {
      eigenA = cscToEigen(*A);
      return 1;
    }

    // Numerical Factorization: perform the LDL^T factorization using Eigen.
    // This replaces the custom ldl_left_simplicial routines.
    int numerical_factorization() {
      solver.compute(eigenA);
      if (solver.info() != Eigen::Success) {
        // Factorization failed – in a real system, report error.
        return 0;
      }
      return 1;
    }

    // Solve the system A*x = rhs.
    // In the original code, solve_only() would perform forward/backward solves
    // on the computed L and D factors. Here, we simply call Eigen's solver.
    double* solve_only() {
      Eigen::Map<Eigen::VectorXd> b(rhs, A->nrow);
      x = solver.solve(b);
      // Allocate a new array for the solution (caller must free).
      double* sol = new double[A->ncol];
      for (int i = 0; i < A->ncol; i++) {
        sol[i] = x(i);
      }
      return sol;
    }
  };

} // end namespace nasoq

// -------------------------------
// Arduino-style setup() and loop() functions.
// In setup(), we test our Eigen-only solver on a small 2x2 problem.
void setup() {
  Serial.begin(9600);
  while (!Serial) { delay(10); }
  Serial.println("Starting Eigen-only NASOQ Solver (No Pivoting) Example...");

  // Example: build a 2x2 diagonal matrix H = [2 0; 0 2] in CSC format.
  int sizeH = 2, nnzH = 2;
  int *Hp = new int[sizeH + 1];
  int *Hi = new int[nnzH];
  double *Hx = new double[nnzH];

  Hp[0] = 0; Hp[1] = 1; Hp[2] = 2;
  Hi[0] = 0; Hi[1] = 1;
  Hx[0] = 2.0; Hx[1] = 2.0;

  nasoq::CSC *H = new nasoq::CSC;
  H->nrow = sizeH; H->ncol = sizeH; H->nzmax = nnzH;
  H->p = Hp; H->i = Hi; H->x = Hx;
  H->stype = -1; H->packed = 1; H->sorted = 1; H->xtype = 1;

  // Set up the right-hand side vector q such that H*x = q.
  // For q = [-4, -4], the expected solution is x = [-2, -2].
  double *q = new double[sizeH];
  q[0] = -4; q[1] = -4;

  nasoq::SolverSettings solver(H, q);
  solver.symbolic_analysis();
  if (!solver.numerical_factorization()) {
    Serial.println("Factorization failed!");
    return;
  }
  double *sol = solver.solve_only();
  Serial.print("Solution 1: ");
  for (int i = 0; i < sizeH; i++) {
    Serial.print(sol[i]); Serial.print(" ");
  }
  Serial.println();
  delete[] sol;

  // Test a new right-hand side: for q = [8, 8], expected solution is x = [4, 4].
  q[0] = 8; q[1] = 8;
  sol = solver.solve_only();
  Serial.print("Solution 2: ");
  for (int i = 0; i < sizeH; i++) {
    Serial.print(sol[i]); Serial.print(" ");
  }
  Serial.println();
  delete[] sol;

  delete H;
  delete[] q;
}

void loop() {
  delay(1000);
}
