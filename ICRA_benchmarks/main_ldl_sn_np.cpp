#include <Arduino.h>
#pragma push_macro("min")
#pragma push_macro("max")
#pragma push_macro("abs")
#pragma push_macro("B1")
#pragma push_macro("F")
#undef min
#undef max
#undef abs
#undef B1
#undef F

#include <../include/eigen/Eigen/Dense>
#include <cassert>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

#pragma pop_macro("F")
#pragma pop_macro("B1")
#pragma pop_macro("abs")
#pragma pop_macro("max")
#pragma pop_macro("min")

namespace nasoq {

  // --------------------------------------------------------------------------
  // CSC structure
  struct CSC {
    int ncol;      // number of columns
    int nrow;      // number of rows
    size_t nzmax;  // max number of nonzeros
    int *p;        // column pointer array (size: ncol+1)
    int *i;        // row indices (size: nzmax)
    double *x;     // numerical values (size: nzmax)
    int stype;     // symmetry type (-1 means unsymmetric)
    int packed;    // flag: 1 if stored in packed form
    int sorted;    // flag: 1 if row indices are sorted
    int xtype;     // type indicator (1 for real)
  };

  // --------------------------------------------------------------------------
  // row_reordering (placeholder): In the original, this reorders rows post-
  // pivoting. Here, we leave it as a no-op.
  static void row_reordering(int /*supNo*/, size_t* /*lC*/, int* /*Li_ptr*/,
                             int* /*lR*/, int* /*blockSet*/, int* /*atree_sm*/,
                             int* /*cT*/, int* /*rT*/, int* /*col2Sup*/,
                             double* /*lValues*/, std::vector<int> /*perm_req*/,
                             int* /*swap_full*/, int* /*xi*/,
                             int* /*map*/, int* /*ws*/, double* /*contribs*/)
  {
    // NOTE: Not implemented yet.
  }

  // --------------------------------------------------------------------------
  // ereach_sn (stub): Originally used for symbolic elimination reach.
  // For now, always return 0 (i.e. no overlap with previous supernodes).
  static int ereach_sn(int /*supNo*/, int* /*cT*/, int* /*rT*/,
                       int /*curCol*/, int /*nxtCol*/, int* /*col2Sup*/,
                       int* /*aTree*/, int* xi, int* /*xi2*/)
  {
    return 0;
  }

  // --------------------------------------------------------------------------
  // ptranspose: Naive CSC matrix transpose.
  static CSC* ptranspose(CSC* A, int /*dummy*/, const int* /*perm*/,
                         void* /*workspace*/, int /*dosymm*/, int &status)
  {
    if (!A) { status = -1; return nullptr; }
    CSC* T = new CSC;
    int m = A->nrow, n = A->ncol;
    T->nrow = n;
    T->ncol = m;
    T->stype = -1;
    T->xtype = 1;
    T->nzmax = A->nzmax;
    T->packed = 1;
    T->sorted = 1;
    T->p = new int[T->ncol+1](); // initialize with zeros
    T->i = new int[T->nzmax];
    T->x = new double[T->nzmax];

    // Count nonzeros per row of A (which become column counts in T).
    for (size_t j = 0; j < A->nzmax; j++) {
      int row = A->i[j];
      if (row >= 0 && row < T->ncol)
        T->p[row]++;
    }
    // Convert counts to starting positions (prefix sum).
    for (int i = 0, sum = 0; i < T->ncol; i++) {
      int tmp = T->p[i];
      T->p[i] = sum;
      sum += tmp;
    }
    T->p[T->ncol] = A->nzmax;
    std::vector<int> current(T->ncol, 0);
    for (int col = 0; col < A->ncol; col++) {
      for (int j = A->p[col]; j < A->p[col+1]; j++) {
        int row = A->i[j];
        double val = A->x[j];
        int pos = T->p[row] + current[row];
        T->i[pos] = col;
        T->x[pos] = val;
        current[row]++;
      }
    }
    status = 0;
    return T;
  }

  // --------------------------------------------------------------------------
  // blocked_2by2_mult: Naive 2x2 block multiplication substitute.
  // Multiplies each column of src by the corresponding diagonal entry from D.
  static void blocked_2by2_mult(int supWdt, int nSupRs,
                                const double *D, const double *src,
                                double *dst, int ld_src, int /*n*/)
  {
    for (int c = 0; c < supWdt; c++) {
      double diagVal = D[c];
      for (int r = 0; r < nSupRs; r++) {
        dst[r + c * nSupRs] = src[r + c * ld_src] * diagVal;
      }
    }
  }

  // --------------------------------------------------------------------------
  // blocked_2by2_solver: Naive solver for the sub-block.
  static void blocked_2by2_solver(int supWdt, const double *D,
                                  double *cur, int rowNo, int ld_cur, int /*n*/)
  {
    for (int c = 0; c < supWdt; c++) {
      double dval = D[c];
      if (fabs(dval) < 1e-14) dval = 1e-14;
      for (int r = 0; r < rowNo; r++) {
        cur[c + (r + supWdt) * ld_cur] /= dval;
      }
    }
  }

  // --------------------------------------------------------------------------
  // reorder_after_sytrf: Naive pivot reordering (simply copies ipiv).
  static int reorder_after_sytrf(int supWdt, double* /*cur*/, int /*nSupR*/,
                                 int* ipiv, int *pivOut, double* /*D*/,
                                 int /*n*/, int* /*swap_full*/, int* /*ws*/)
  {
    int is_perm = 0;
    for (int i = 0; i < supWdt; i++) {
      pivOut[i] = ipiv[i];
      if (ipiv[i] != i) is_perm = 1;
    }
    return is_perm;
  }

  // --------------------------------------------------------------------------
  // my_dlapmt: Minimal replacement for LAPACKE_dlapmt; here, does nothing.
  static int my_dlapmt(int /*fwd*/, int /*M*/, int /*N*/,
                       double* /*A*/, int /*lda*/, int* /*piv*/)
  {
    return 0;
  }

  // --------------------------------------------------------------------------
  // eigen_dgemm: Replaces DGEMM using Eigen.
  static void eigen_dgemm(const char* /*transA*/, const char* /*transB*/,
                          const int *M, const int *N, const int *K,
                          const double *alpha,
                          const double *A, const int *lda,
                          const double *B, const int *ldb,
                          const double *beta,
                          double *C, const int *ldc)
  {
    int m = *M, n = *N, k = *K;
    double alp = *alpha, bet = *beta;
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >
        matA(A, m, k);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >
        matB(B, *ldb, n);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >
        matC(C, m, n);
    matC = alp * (matA * matB.transpose()) + bet * matC;
  }

  // --------------------------------------------------------------------------
  // eigen_dtrsm: Replaces DTRSM using Eigen.
  static void eigen_dtrsm(const char* /*side*/, const char* /*uplo*/,
                          const char* /*transA*/, const char* /*diag*/,
                          const int *M, const int *N,
                          const double *alpha,
                          const double *A, const int *lda,
                          double *B, const int *ldb)
  {
    int m = *M, n = *N;
    double alp = *alpha;
    // Map A as an n x n matrix.
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >
        matA(A, *lda, n);
    // Map B as an m x n matrix.
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >
        matB(B, *ldb, n);
    // Solve X * A^T = B => X = B * (A^T)^{-1}.
    Eigen::MatrixXd Atri = matA.topRows(n);
    Eigen::MatrixXd X = matB * Atri.transpose().triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(n, n));
    X *= alp;
    matB = X;
  }

  // --------------------------------------------------------------------------
  // ldl_left_sn_02_v2: Main factorization routine.
  // NOTE: Heavy routines such as DGEMM, DTRSM, and LDLT factorization are 
  //       replaced by our eigen_dgemm, eigen_dtrsm, and Eigen::LDLT respectively.
  //       Other functions (ereach_sn, row_reordering, 2x2 block ops) are 
  //       provided in a minimal/naive form.
  bool ldl_left_sn_02_v2(int n, int *c, int *r, double *values,
                         size_t *lC, int *lR, size_t *Li_ptr, double *lValues,
                         double *D, int *blockSet, int supNo, double *timing,
                         int *aTree, int *cT, int *rT, int *col2Sup,
                         int super_max, int col_max, int &nbpivot,
                         int *perm_piv, int *atree_sm, double threshold)
  {
    (void)timing;
    (void)threshold;
    (void)atree_sm;
    int *xi = new int[3 * supNo]();
    int *swap_full = new int[n]();
    std::vector<int> perm_req;
    int *map = new int[n]();
    double *contribs = new double[super_max * col_max]();
    double *trn_diag = new double[super_max * col_max]();
    int *ws = new int[3 * super_max];
    int *ipiv = new int[super_max]();
    int top = 0;
    // Removed unused variable "info"
    double one[2] = {1.0, 0.0}, zero[2] = {0.0, 0.0};

    nbpivot = 0;
    for (int s = 1; s <= supNo; ++s) {
      int curCol = (s != 0 ? blockSet[s - 1] : 0);
      int nxtCol = blockSet[s];
      int supWdt = nxtCol - curCol;
      int nSupR = static_cast<int>(Li_ptr[nxtCol] - Li_ptr[curCol]);
      for (int i = static_cast<int>(Li_ptr[curCol]), cnt = 0; i < static_cast<int>(Li_ptr[nxtCol]); ++i) {
        map[lR[i]] = cnt++;
      }
      for (int i = curCol; i < nxtCol; ++i) {
        // Removed unused "pad"
        for (int j = c[i]; j < c[i + 1]; ++j) {
          lValues[lC[i] + map[r[j]]] = values[j];
        }
      }
      top = ereach_sn(supNo, cT, rT, curCol, nxtCol, col2Sup, aTree, xi, xi + supNo);
      assert(top >= 0);
      for (int i = top; i < supNo; ++i) {
        int lSN = xi[i];
        int cSN = blockSet[lSN];
        int cNSN = blockSet[lSN + 1];
        int nSNRCur = static_cast<int>(Li_ptr[cNSN] - Li_ptr[cSN]);
        int supWdts = cNSN - cSN;
        int lb = 0, ub = 0;
        bool sw = true;
        for (int j = static_cast<int>(Li_ptr[cSN]); j < static_cast<int>(Li_ptr[cNSN]); ++j) {
          if (lR[j] >= curCol && sw) {
            lb = j - Li_ptr[cSN];
            sw = false;
          }
          if (lR[j] < curCol + supWdt && !sw) {
            ub = j - Li_ptr[cSN];
          }
        }
        int localSupRows = nSNRCur - lb;
        int ndrow1 = (ub - lb + 1);
        const double *src = &lValues[lC[cSN] + lb];
        blocked_2by2_mult(supWdts, localSupRows, &D[cSN], src, trn_diag, nSNRCur, n);
        {
          int M = localSupRows, N = ndrow1, K = supWdts;
          eigen_dgemm("N", "C", &M, &N, &K, one,
                      trn_diag, &M,
                      src, &nSNRCur,
                      zero,
                      contribs, &M);
        }
        for (int i2 = 0; i2 < ndrow1; ++i2) {
          int col = map[lR[Li_ptr[cSN] + i2 + lb]];
          for (int j2 = i2; j2 < localSupRows; ++j2) {
            int cRow = lR[Li_ptr[cSN] + j2 + lb];
            lValues[lC[curCol + col] + map[cRow]] -= contribs[i2 * localSupRows + j2];
          }
        }
      } // End contributions for current supernode

      // --- Factor the current diagonal block ---
      {
        double *cur = &lValues[lC[curCol]];
        // Map the current supernode block as an Eigen matrix
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >
          curBlock(cur, nSupR, supWdt);
        // Use Eigen's LDLT factorization on the current block
        Eigen::LDLT<Eigen::MatrixXd> ldlt(curBlock);
        ldlt.compute(curBlock);
        for (int i = 0; i < supWdt; ++i) { ipiv[i] = i; }  // assume no pivoting changes
        int is_perm = reorder_after_sytrf(supWdt, cur, nSupR, ipiv,
                                           &perm_piv[curCol], &D[curCol], n,
                                           swap_full, ws);
        if (is_perm) {
          int rowNo = nSupR - supWdt;
          my_dlapmt(1, rowNo, supWdt, &cur[supWdt], nSupR, &perm_piv[curCol]);
          perm_req.push_back(s);
        }
        for (int m = 0; m < supWdt; ++m) {
          perm_piv[curCol + m] += (curCol - 1);
        }
        // Extract the diagonal from the factorization.
        Eigen::VectorXd diag = ldlt.vectorD();
        for (int l = 0; l < supWdt; ++l) {
          D[curCol + l] = diag(l);
          cur[l + l * nSupR] = 1.0; // set unit diagonal
        }
        {
          int rowNo = nSupR - supWdt;
          eigen_dtrsm("R", "L", "C", "U", &rowNo, &supWdt, one,
                      cur, &nSupR, &cur[supWdt], &nSupR);
          blocked_2by2_solver(supWdt, &D[curCol], &cur[supWdt], rowNo, nSupR, n);
        }
      }
    } // End loop over supernodes

    // Call row_reordering placeholder (naive no-op)
    row_reordering(supNo, lC, reinterpret_cast<int*>(Li_ptr), lR, blockSet, nullptr,
                   cT, rT, col2Sup, lValues, perm_req, swap_full, xi, map, ws, contribs);

    delete[] contribs;
    delete[] trn_diag;
    delete[] xi;
    delete[] map;
    delete[] ws;
    delete[] swap_full;
    delete[] ipiv;
    return true;
  }

  // --------------------------------------------------------------------------
  // allocateAC: Allocates or frees a CSC matrix.
  static void allocateAC(CSC* &A, int nrow, int ncol, int nnz, bool alloc)
  {
    if (!A) return;
    if (!alloc || nrow == 0 || ncol == 0 || nnz == 0) {
      if (A->p) { delete [] A->p; A->p = nullptr; }
      if (A->i) { delete [] A->i; A->i = nullptr; }
      if (A->x) { delete [] A->x; A->x = nullptr; }
      delete A; A = nullptr;
      return;
    }
    A->ncol = ncol;
    A->nrow = nrow;
    A->nzmax = nnz;
    A->p = new int[ncol+1];
    A->i = new int[nnz];
    A->x = new double[nnz];
    A->packed = 1;
    A->sorted = 1;
    A->stype = -1;
    A->xtype = 1;
  }

  // --------------------------------------------------------------------------
  // profiling_solver_info: Timing and profiling info using std::chrono.
  struct profiling_solver_info {
    double fact_time, analysis_time, solve_time, iter_time;
    double ordering_time, update_time, piv_reord;
    double *timing_chol;
    profiling_solver_info(int nt) : fact_time(0), analysis_time(0), solve_time(0),
      iter_time(0), ordering_time(0), update_time(0), piv_reord(0) {
        timing_chol = new double[4 + nt]();
    }
    ~profiling_solver_info() { delete[] timing_chol; }
    std::chrono::time_point<std::chrono::system_clock> tic() {
        return std::chrono::system_clock::now();
    }
    std::chrono::time_point<std::chrono::system_clock> toc() {
        return std::chrono::system_clock::now();
    }
    double elapsed_time(std::chrono::time_point<std::chrono::system_clock> beg,
                        std::chrono::time_point<std::chrono::system_clock> lst) {
        return std::chrono::duration_cast<std::chrono::duration<double>>(lst - beg).count();
    }
    void print_profiling() {
        std::cout << "analysis time: " << analysis_time << ";";
        std::cout << "fact time: " << fact_time << ";";
        std::cout << "update time: " << update_time << ";";
        std::cout << "reordering pivot time: " << piv_reord << ";";
        std::cout << "solve time: " << solve_time << ";";
    }
  };

  // --------------------------------------------------------------------------
  // LFactor structure: holds the factorization workspace.
  // NOTE: Internally, we use size_t for the pointer array "p" (lC) and for i_ptr.
  struct LFactor {
    int nsuper;
    size_t xsize;
    int nzmax;
    size_t *p;         // Column pointer array (lC)
    int *s;            // Row indices (lR)
    size_t *i_ptr;     // Additional pointer array (Li_ptr)
    double *x;         // Factor values (lValues)
    double *x_s;       // For simplicial factor (unused here)
    int *Parent;
    int *sParent;
    int *Perm;
    int *IPerm;
    int *col2Sup;
    int *super;
  };

  // --------------------------------------------------------------------------
  // SolverSettings class: Wraps the entire solver state.
  // This mimics the NASOQ interface but uses our Eigen-based factorization.
  class SolverSettings {
  public:
    CSC *A;          // Input matrix (CSC format)
    double *rhs;     // Right-hand side vector
    CSC *A_ord;      // Shallow copy of A after reordering
    CSC *AT_ord;     // Transpose of A_ord
    CSC *SM;         // Supernode matrix (if applicable)
    profiling_solver_info *psi;
    LFactor *L;      // Factorization workspace/info
    double *valL;    // Pointer to factor values (lValues)
    double *d_val;   // Diagonal D from factorization
    int num_pivot;
    int ldl_variant;
    int solver_mode;
    double reg_diag;
    double *x;       // Solution vector

    // Constructor: initialize members with minimal defaults.
    SolverSettings(CSC *Amat, double *rhs_in) {
      default_setting();
      A = Amat;
      rhs = rhs_in;
      A_ord = nullptr;
      AT_ord = nullptr;
      SM = new CSC;
      psi = new profiling_solver_info(1);
      solver_mode = 0;
      reg_diag = 1e-9;
      num_pivot = 0;
      int n = A->ncol;
      x = new double[n];
      // Initialize L factor with a single supernode assumption.
      L = new LFactor;
      L->nsuper = 1;
      L->xsize = 0;
      L->nzmax = A->nzmax;
      L->p = nullptr;
      L->s = nullptr;
      L->i_ptr = nullptr;
      L->x = nullptr;
      valL = nullptr;
      d_val = nullptr;
    }
    ~SolverSettings(){
      delete psi;
      if (A_ord) allocateAC(A_ord, 0, 0, 0, false);
      if (AT_ord) allocateAC(AT_ord, 0, 0, 0, false);
      if (L) {
        if(L->p) { delete[] L->p; }
        if(L->s) { delete[] L->s; }
        if(L->i_ptr) { delete[] L->i_ptr; }
        if(L->x) { delete[] L->x; }
        delete L;
      }
      delete SM;
      delete[] x;
      if(d_val) delete[] d_val;
    }
    void default_setting() {
      ldl_variant = 2;
      num_pivot = 0;
    }
    // symbolic_analysis: Makes a shallow copy of A and builds AT_ord; allocates L factor arrays.
    int symbolic_analysis() {
      int n = A->ncol;
      A_ord = new CSC;
      *A_ord = *A;  // Shallow copy
      int status = 0;
      AT_ord = ptranspose(A, 2, nullptr, nullptr, 0, status);
      // Allocate workspace for the factor.
      L->p = new size_t[n+1];
      L->s = new int[A->nzmax];
      L->i_ptr = new size_t[n+1];
      L->x = new double[A->nzmax];
      valL = L->x;
      d_val = new double[2*n];
      for (int i = 0; i <= n; i++) {
        L->p[i] = 0;
        L->i_ptr[i] = 0;
      }
      for (size_t j = 0; j < A->nzmax; j++) {
        L->s[j] = 0;
        L->x[j] = 0.0;
      }
      for (int i = 0; i < 2*n; i++) { d_val[i] = 0.0; }
      return 1;
    }
    // numerical_factorization: Calls our Eigen-based factorization routine.
    int numerical_factorization() {
      int supNo = 1;
      int blockSet[2] = {0, A->ncol};
      bool ok = ldl_left_sn_02_v2(A->ncol, A->p, A->i, A->x,
                                  L->p, L->s, L->i_ptr, L->x, d_val,
                                  blockSet, supNo, nullptr,
                                  nullptr, nullptr, nullptr, nullptr,
                                  64, 64, num_pivot, nullptr, nullptr, reg_diag);
      return ok ? 1 : 0;
    }
    // solve_only: Naively computes x = rhs / d_val.
    double* solve_only() {
      int n = A->ncol;
      for (int i = 0; i < n; i++) {
        if (fabs(d_val[i]) < 1e-14) d_val[i] = 1e-14;
        x[i] = rhs[i] / d_val[i];
      }
      return x;
    }
    // reorder_matrix: Calls our row_reordering placeholder.
    void reorder_matrix() {
      row_reordering(0, L->p, reinterpret_cast<int*>(L->i_ptr), L->s,
                     nullptr, nullptr, nullptr, nullptr, nullptr,
                     nullptr, std::vector<int>(), nullptr, nullptr, nullptr, nullptr, nullptr);
    }
  };

} // end namespace nasoq

// ============================================================================
// main() - Testing the solver with a small 2x2 system.
int main(){
  // Build a 2x2 diagonal matrix H = [2 0; 0 2] in CSC format.
  int sizeH = 2, nnzH = 2;
  double *q = new double[sizeH];
  int *Hp = new int[sizeH+1];
  int *Hi = new int[nnzH];
  double *Hx = new double[nnzH];

  q[0] = -4; q[1] = -4;
  Hp[0] = 0; Hp[1] = 1; Hp[2] = 2;
  Hi[0] = 0; Hi[1] = 1;
  Hx[0] = 2; Hx[1] = 2;

  nasoq::CSC *H = new nasoq::CSC;
  H->ncol = sizeH;
  H->nrow = sizeH;
  H->nzmax = nnzH;
  H->p = Hp;
  H->i = Hi;
  H->x = Hx;
  H->stype = -1;
  H->packed = 1;
  H->sorted = 1;
  H->xtype = 1;

  // Set up the solver.
  nasoq::SolverSettings *lbl = new nasoq::SolverSettings(H, q);
  lbl->ldl_variant = 2;
  lbl->solver_mode = 0;
  lbl->reg_diag = pow(10, -9);

  lbl->symbolic_analysis();
  lbl->numerical_factorization();
  double *x = lbl->solve_only();

  Serial.begin(9600);
  while (!Serial) { delay(10); }
  Serial.print("Solution 1: ");
  for (int i = 0; i < sizeH; ++i) {
      Serial.print(x[i]);
      Serial.print(" ");
  }
  Serial.println();

  // Update rhs and resolve: New rhs [8, 8] should yield solution [4, 4].
  double *new_q = new double[sizeH];
  new_q[0] = 8; new_q[1] = 8;
  for (int i = 0; i < sizeH; i++) {
      lbl->rhs[i] = new_q[i];
  }
  x = lbl->solve_only();
  Serial.print("Solution 2: ");
  for (int i = 0; i < sizeH; i++) {
      Serial.print(x[i]);
      Serial.print(" ");
  }
  Serial.println();

  delete lbl;
  // NOTE: H->p, H->i, and H->x were not allocated on the heap by our code.
  delete H;
  delete[] q;
  delete[] new_q;
  return 0;
}

void setup()
{
  srand(123);
  // initialize LED digital pin as an output.
  pinMode(LED_BUILTIN, OUTPUT);

  // start serial terminal
  Serial.begin(9600);

  delay(15000);

  while (!Serial)
  { // wait to connect
    continue;
  }

  // Serial.println("Start");

  Serial.println("Start NASOQ-MPC...");

  // main();
}

void loop()
{
}
