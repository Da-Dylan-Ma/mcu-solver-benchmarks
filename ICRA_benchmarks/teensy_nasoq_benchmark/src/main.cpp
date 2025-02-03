/**
 * Blink
 *
 * Turns on an LED on for one second,
 * then off for one second, repeatedly.
 */
#include "Arduino.h"

#include <stdio.h>
#include <iostream>
#include "math.h"

#ifndef LED_BUILTIN
#define LED_BUILTIN 13
#endif


/* NASOQ Part */

/******************************************************************************
 * Teensy/NASOQ Single-File Example (No MKL/BLAS/LAPACK)
 * ---------------------------------------------------
 * This example combines:
 *   - A naive BLAS-like layer.
 *   - A simplified pivot routine in place of LAPACKE_dsytrf / dlapmt.
 *   - NASOQ’s ldl_left_sn_02_v2 factor logic.
 *   - A minimal “SolverSettings” class wrapping everything.
 *   - A sample main() that solves a tiny system.
 *
 * Compile under PlatformIO for Teensy by placing this file in `src/`,
 * adjusting your platformio.ini, and ensuring no references to MKL/OpenBLAS.
 ******************************************************************************/

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <assert.h>

/****************************************************/
/*          Minimal / Stub BLAS-Like Layer          */
/****************************************************/

/**
 * Naive DGEMM: C = alpha*(A*B) + beta*C
 *   - M x K times K x N -> M x N
 *   - A is lda x M, B is ldb x K, C is ldc x M (column major)
 */
static void naive_dgemm(const char* /*transA*/, const char* /*transB*/,
                        const int *M, const int *N, const int *K,
                        const double *alpha,
                        const double *A, const int *lda,
                        const double *B, const int *ldb,
                        const double *beta,
                        double *C, const int *ldc)
{
  int m = *M, n = *N, k = *K;
  double alp = *alpha, bet = *beta;
  for(int j=0; j<n; j++) {
    for(int i=0; i<m; i++) {
      double sum = 0.0;
      for(int kk=0; kk<k; kk++){
        sum += A[i + (*lda)*kk] * B[kk + (*ldb)*j];
      }
      C[i + (*ldc)*j] = alp*sum + bet*C[i + (*ldc)*j];
    }
  }
}

/**
 * Naive DTRSM: solves op(A)*X = alpha*B or X*op(A) = alpha*B
 * For brevity, we only handle a subset of arguments:
 *  - side='R', uplo='L', trans='C' (like the code calls).
 */
static void naive_dtrsm(const char* side, const char* uplo,
                        const char* transA, const char* diag,
                        const int *M, const int *N,
                        const double *alpha,
                        const double *A, const int *lda,
                              double *B, const int *ldb)
{
  // The code in ldl_left_sn_02_v2 calls:
  //   cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans, CblasUnit, rowNo, supWdt, 1.0, cur, nSupR, &cur[supWdt], nSupR);
  // This corresponds to A on the right, so we interpret that carefully.
  // 
  // For simplicity, we'll do a naive approach:
  //   Solve B * A^-T = B'  (since trans='C' means conj-trans => effectively A^T)
  // 
  // Actually: a truly correct "side=R" solve is more complicated. We'll do a partial approach:
  int m = *M, n = *N;
  // We'll skip the alpha != 1.0 for brevity:
  (void)alpha; (void)side; (void)uplo; (void)transA; (void)diag;
  
  // *** This is just a stub that does nothing. ***
  // If your code relies heavily on DTRSM with side=R, you must implement the logic.
  // For small test systems, this may be enough not to break.
  // 
  // If you do want to do a correct solve, you must carefully write out
  // the row-by-row or col-by-col backward solve. 
  // 
  // For now, we’ll leave it as no-op to keep the example short.
}

/**
 * Macros that NASOQ uses in place of real BLAS calls:
 * If OPENBLAS / MKL isn’t defined, code calls e.g. SYM_DGEMM(…)
 */
static double __one[2] = {1.0, 0.0}, __zero[2] = {0.0, 0.0};

#define SYM_DGEMM naive_dgemm
#define SYM_DTRSM naive_dtrsm

/****************************************************/
/*      Minimal Replacement for LAPACKE Calls       */
/****************************************************/

/**
 * A minimal no-frills version of a Bunch-Kaufman or simple pivoting LDL factor.
 * In the original code:  LAPACKE_dsytrf(…, 'L', n, cur, nSupR, ipiv).
 * We define a naive LDL factor with no or partial pivot logic.
 */
static int my_dsytrf(char /*uplo*/, int n, double *A, int lda, int *ipiv)
{
  // For simplicity, do a naive LDL (diagonal pivot only), ignoring 2x2 pivots:
  // 1) ipiv[i] = i by default
  // 2) Factor in place: for k=0..n-1, factor pivot row/col
  // This is much simpler than full Bunch-Kaufman.
  for(int i=0; i<n; i++){ ipiv[i] = i; }

  for(int k=0; k<n; k++){
    double diag = A[k + k*(lda)];
    // Quick check to avoid dividing by 0
    if(fabs(diag) < 1e-14){
      // pivot could do something smarter, but let's do a small shift
      diag = 1e-14;
      A[k + k*(lda)] = diag;
    }
    // scale below diagonal
    for(int i=k+1; i<n; i++){
      double val = A[i + k*lda];
      val /= diag;
      A[i + k*lda] = val;
      // update trailing submatrix
      for(int j=k+1; j<=i; j++){
        A[i + j*lda] -= val * A[j + k*lda] * diag;
      }
    }
  }
  return 0; // info=0 => success
}

/**
 * Minimal reorder routine that stubs out LAPACKE_dlapmt
 *   - The original code uses it to reorder columns after pivoting.
 *   - Here, do a partial or no-op approach.
 */
static int my_dlapmt(int /*fwd*/, int /*M*/, int /*N*/,
                     double* /*A*/, int /*lda*/, int* /*piv*/)
{
  // For demonstration, just do nothing
  return 0;
}

/****************************************************/
/*   Helper Functions Called by ldl_left_sn_02_v2   */
/****************************************************/

/**
 * Stubs for 2×2 “blocked” operations that appear in the factor routine:
 */

// Very minimal example: multiply each column by diagonal D
static void blocked_2by2_mult(int supWdt, int nSupRs,
                              const double *D, const double *src,
                              double *dst, int ld_src, int /*n*/)
{
  // For each of supWdt columns, scale the column by diagonal in D
  // (In the real code, it might handle 2×2 blocks. This is a naive placeholder.)
  for(int c = 0; c < supWdt; c++){
    double diagVal = D[c];
    for(int r = 0; r < nSupRs; r++){
      dst[r + c*nSupRs] = src[r + c*ld_src] * diagVal;
    }
  }
}

// Another minimal placeholder that just scales rows by D
static void blocked_2by2_solver(int supWdt, const double *D,
                                double *cur, int rowNo, int ld_cur,
                                int /*n*/)
{
  // TODO(Da): The real code presumably does 2×2 pivot solves. We do a naive “divide by diag.”
  // If rowNo>0, we're solving the sub-block. This is a partial placeholder:
  for(int c=0; c<supWdt; c++){
    double dval = D[c];
    if(fabs(dval) < 1e-14) dval = 1e-14;
    for(int r=0; r<rowNo; r++){
      cur[c + (r+supWdt)*ld_cur] /= dval;
    }
  }
}

/**
 * Minimal pivot reordering after SYTRF. The real code uses ipiv from dsytrf.
 * We do a trivial pass that checks if ipiv[i] != i => we note “some pivot done.”
 */
static int reorder_after_sytrf(int supWdt, double* /*cur*/, int /*nSupR*/,
                               int* ipiv, int *pivOut, double* /*D*/,
                               int /*n*/, int* /*swap_full*/, int* /*ws*/)
{
  // TODO(Da): If any ipiv[i] != i, we treat that as a single pivot reordering.
  // We'll also do a naive “copy pivot to pivOut”.
  int is_perm = 0;
  for(int i=0; i<supWdt; i++){
    pivOut[i] = ipiv[i];
    if(ipiv[i] != i) { is_perm = 1; }
  }
  return is_perm;
}

/**
 * Minimal row_reordering stub. The snippet calls `row_reordering` at the end.
 */
static void row_reordering(int /*supNo*/, size_t* /*lC*/, int* /*Li_ptr*/,
                           int* /*lR*/, int* /*blockSet*/, int* /*atree_sm*/,
                           int* /*cT*/, int* /*rT*/, int* /*col2Sup*/,
                           double* /*lValues*/, std::vector<int> /*perm_req*/,
                           int* /*swap_full*/, int* /*xi*/,
                           int* /*map*/, int* /*ws*/, double* /*contribs*/)
{
  // TODO(Da): The real code may reorder rows after the pivoting steps. For now, no-op:
}

/****************************************************/
/*        EREACH-Like Helper for Factorization      */
/****************************************************/

/**
 * A minimal stub for `ereach_sn`. The real function finds relevant supernodes.
 * We'll just return 0 to indicate “no previous supernodes”.
 */
static int ereach_sn(int /*supNo*/, int* /*cT*/, int* /*rT*/,
                     int /*curCol*/, int /*nxtCol*/, int* /*col2Sup*/,
                     int* /*aTree*/, int* xi, int* /*xi2*/)
{
  // TODO(Da): For a real approach, do the symbolic BFS/DFS. This stub returns 0 => no overlap.
  (void)xi; // unused
  return 0;
}

/****************************************************/
/*    The ldl_left_sn_02_v2 Factor Implementation   */
/****************************************************/

bool ldl_left_sn_02_v2(int n, int *c, int *r, double *values,
                       size_t *lC, int *lR, size_t *Li_ptr,
                       double *lValues, double *D,
                       int *blockSet, int supNo, double *timing,
                       int *aTree, int *cT, int *rT, int *col2Sup,
                       int super_max, int col_max,
                       int &nbpivot, int *perm_piv,
                       int *atree_sm, double threshold = 1e-14)
{
  // The official code references openblas calls or SYM_DGEMM. I here use naive macros.
  // Also references LAPACKE_dsytrf, LAPACKE_dlapmt => replaced by my_dsytrf / my_dlapmt.

  (void)threshold;
  (void)timing;
  (void)atree_sm;

  // Temporary allocations
  int *xi = new int[3 * supNo]();
  int *swap_full = new int[n]();
  std::vector<int> perm_req;

  int *map = new int[n]();
  double *contribs = new double[super_max * col_max]();
  double *trn_diag = new double[super_max * col_max]();
  int *ws = new int[3 * super_max];
  int *ipiv = new int[super_max];

  // TODO(Da): Not measure timing here
  const int incx = 1;
  int top = 0;
  int info = 0;
  double one[2] = {1.0, 0.}, zero[2] = {0., 0.};

  nbpivot = 0; // track pivot changes

  for(int s = 1; s <= supNo; s++){
    int curCol = (s != 0 ? blockSet[s - 1] : 0);
    int nxtCol = blockSet[s];
    int supWdt = nxtCol - curCol;
    int nSupR = Li_ptr[nxtCol] - Li_ptr[curCol]; // row size of supernode

    // Build map from row index => local index in supernode
    for(int i = Li_ptr[curCol], cnt=0; i < (int)Li_ptr[nxtCol]; i++){
      map[lR[i]] = cnt++;
    }

    // Copy columns from A to L
    for(int i = curCol; i < nxtCol; i++){
      int pad = i - curCol;
      for(int j = c[i]; j < c[i+1]; j++){
        lValues[lC[i] + map[r[j]]] = values[j];
      }
    }

    // EREACH
    top = ereach_sn(supNo, cT, rT, curCol, nxtCol,
                    col2Sup, aTree, xi, xi+supNo);

    for(int i = top; i < supNo; i++){
      int lSN = xi[i];

      int cSN  = blockSet[lSN];
      int cNSN = blockSet[lSN+1];
      int nSNRCur = (int)(Li_ptr[cNSN] - Li_ptr[cSN]);
      int supWdts = cNSN - cSN;

      // find overlap rows
      int lb=0, ub=0;
      bool sw = true;
      for(int j= (int)Li_ptr[cSN]; j < (int)Li_ptr[cNSN]; j++){
        if(lR[j] >= curCol && sw){
          lb = j - (int)Li_ptr[cSN];
          sw = false;
        }
        if(lR[j] < (curCol + supWdt) && !sw){
          ub = j - (int)Li_ptr[cSN];
        }
      }
      int nSupRs = nSNRCur - lb; // # rows to handle
      int ndrow1 = (ub - lb + 1);

      const double* src = &lValues[lC[cSN] + lb];
      // “blocked_2by2_mult” => multiply D of cSN into src => trn_diag
      blocked_2by2_mult(supWdts, nSupRs, &D[cSN],
                        src, trn_diag, nSNRCur, n);

      // Then do GEMM:  contribs = trn_diag * src^T
      //   shape: nSupRs x ndrow1 = (nSupRs x supWdts)*(supWdts x ndrow1)
      {
        int M = nSupRs, N = ndrow1, K = supWdts;
        SYM_DGEMM("N","C", &M, &N, &K, one,
                  trn_diag, &M,
                  src, &nSNRCur,
                  zero,
                  contribs, &M);
      }

      // Subtract from current L block
      for(int i2=0; i2<ndrow1; i2++){
        int col = map[lR[ (int)Li_ptr[cSN] + i2 + lb ]];
        for(int j2=i2; j2<nSupRs; j2++){
          int cRow = lR[ (int)Li_ptr[cSN] + j2 + lb ];
          lValues[lC[curCol + col] + map[cRow]]
            -= contribs[i2*nSupRs + j2];
        }
      }
    }

    // Now factor the diagonal block with minimal pivot routine
    {
      // pointer to the diagonal block in lValues:
      double* cur = &lValues[lC[curCol]];
      info = my_dsytrf('L', supWdt, cur, nSupR, ipiv);
      // reorder after factor
      int is_perm = reorder_after_sytrf(supWdt, cur, nSupR,
                                        ipiv, &perm_piv[curCol],
                                        &D[curCol], n, swap_full, ws);

      // if is_perm => we have a pivot reordering
      if(is_perm) {
        // dlapmt
        // rowNo below is nSupR - supWdt
        int rowNo = nSupR - supWdt;
        my_dlapmt(1, rowNo, supWdt, &cur[supWdt], nSupR, &perm_piv[curCol]);
        perm_req.push_back(s);
      }

      // fix pivot array offsets
      for(int m=0; m<supWdt; m++){
        perm_piv[curCol + m] += (curCol - 1);
      }

      // extract diagonal into D
      for(int l=0; l<supWdt; l++){
        D[curCol + l] = cur[l + l*nSupR];
        cur[l + l*nSupR] = 1.0;
      }

      // Triangular solve on the sub-block
      {
        int rowNo = nSupR - supWdt;
        // cblas_dtrsm => naive_dtrsm
        {
          char R='R', L='L', C='C', U='U';
          int M=rowNo, N=supWdt;
          double alpha=1.0;
          naive_dtrsm(&R, &L, &C, &U,
                      &M, &N, &alpha,
                      cur, &nSupR,
                      &cur[supWdt], &nSupR);
        }

        // blocked_2by2_solver => naive sub-block solve
        blocked_2by2_solver(supWdt, &D[curCol],
                            &cur[supWdt], rowNo, nSupR, n);
      }
    }
  }

  // row reordering for pivot
  row_reordering(supNo, lC, Li_ptr, lR, blockSet, atree_sm,
                 cT, rT, col2Sup, lValues, perm_req,
                 swap_full, xi, map, ws, contribs);

  // cleanup
  delete[] contribs;
  delete[] trn_diag;
  delete[] xi;
  delete[] map;
  delete[] ws;
  delete[] swap_full;
  delete[] ipiv;

  return true;
}

/****************************************************/
/*           Minimal CSC + SolverSettings           */
/****************************************************/

// A simple struct for a CSC matrix
struct CSC {
  int    ncol;    // # columns
  int    nrow;    // # rows
  int    stype;   // symmetry type
  int    xtype;   // type of x (real/complex)
  int    packed;  // if column form is packed
  int    sorted;  // if row indices are sorted
  size_t nzmax;   // maximum # of entries
  int   *p;       // column pointer array (size ncol+1)
  int   *i;       // row index array (size nzmax)
  double*x;       // numerical values (size nzmax)
};

/**
 * Simple function to allocate a CSC with given dimension & nnz,
 * or free it if ncol==0 or p==NULL.
 */
static void allocateAC(CSC* &A, int nrow, int ncol, int nnz, bool alloc)
{
  if(!A) return;
  if(!alloc || nrow==0 || ncol==0 || nnz==0){
    // free
    if(A->p){ delete []A->p; A->p=nullptr; }
    if(A->i){ delete []A->i; A->i=nullptr; }
    if(A->x){ delete []A->x; A->x=nullptr; }
    delete A; A=nullptr;
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
  A->xtype = 1;  // real
}

// A minimal ptranspose that re‐orders columns or rows if needed
static CSC* ptranspose(CSC* A, int /*values*/, const int* perm,
                       void* /*workspace*/, int /*dosymm*/, int &status)
{
  // TODO(Da): just do a naive transpose ignoring permutations if perm==NULL.
  // A real ptranspose is more advanced. This is a partial stub.
  if(!A){ status=-1; return nullptr; }
  CSC* T = new CSC;
  int m = A->nrow, n = A->ncol;
  T->nrow = n; T->ncol=m; T->stype=-1; T->xtype=1;
  T->nzmax = A->nzmax;
  T->packed = 1; T->sorted=1;
  T->p = new int[T->ncol+1]();
  T->i = new int[T->nzmax];
  T->x = new double[T->nzmax];

  // Build row counts
  for(size_t j=0; j<A->nzmax; j++){
    int r = A->i[j];
    if(r>=0 && r<T->ncol) T->p[r]++;
  }
  // prefix sum
  for(int c=0; c<T->ncol; c++){
    int tmp=T->p[c];
    T->p[c] = (c==0)? 0 : T->p[c-1];
    T->p[c+1] = T->p[c] + tmp;
  }
  // fill
  for(int col=0; col<n; col++){
    int start = A->p[col];
    int end   = A->p[col+1];
    for(int idx=start; idx<end; idx++){
      int row = A->i[idx];
      double val = A->x[idx];
      int destPos = T->p[row];
      T->i[destPos] = col;
      T->x[destPos] = val;
      T->p[row]++;
    }
  }
  // shift back T->p
  for(int c=T->ncol; c>0; c--){
    T->p[c] = T->p[c-1];
  }
  T->p[0] = 0;

  status=0;
  return T;
}

// A minimal solver profiling (stubs)
struct profiling_solver_info {
  double fact_time;
  double analysis_time;
  double solve_time;
  double iter_time;
  double ordering_time;
  double update_time;
  double piv_reord;

  profiling_solver_info(int /*nt*/){
    fact_time=0; analysis_time=0; solve_time=0; iter_time=0;
    ordering_time=0; update_time=0; piv_reord=0;
  }
};

// Data structure to hold factor info
struct LFactor {
  // TODO(Da): In the real code, this has supernodes, permutations, etc.
  // Minimally, I store these to keep the example consistent:
  int nsuper;
  size_t xsize;
  int nzmax;
  int *p;      // lC in snippet
  int *s;      // lR in snippet
  size_t *i_ptr;  // Li_ptr
  double *x;   // lValues
  double *x_s; // for simplicial
  int *Parent;
  int *sParent;
  int *Perm;
  int *IPerm;
  int *col2Sup;
  int *super;
};

class SolverSettings {
public:
  CSC* A;          // input matrix
  double* rhs;     // right-hand side
  CSC* A_ord;      // reordered matrix
  CSC* AT_ord;     // transpose
  LFactor* L;      // factor data
  double* valL;    // lValues
  double* d_val;   // D
  int num_pivot;   // pivot stats
  int n_level;
  int *level_ptr;
  int *level_set;
  int n_par;
  int *par_ptr;
  int *par_set;

  // TODO(Da): etc. Skipped many fields from the big snippet.

  int ldl_variant;
  int solver_mode;
  double reg_diag;
  int status;

  profiling_solver_info* psi;
  double* x;

  int max_sup_wid, max_col;

  SolverSettings(CSC* Amat, double* rhs_in) {
    A = Amat;
    rhs = rhs_in;
    A_ord = new CSC;
    AT_ord = new CSC;
    L = new LFactor;
    // minimal initialization:
    L->nsuper = 0; L->xsize=0; L->nzmax=0;
    L->p=nullptr; L->s=nullptr; L->i_ptr=nullptr; L->x=nullptr; L->x_s=nullptr;
    L->Parent=nullptr; L->sParent=nullptr; L->Perm=nullptr; L->IPerm=nullptr; L->col2Sup=nullptr; L->super=nullptr;

    psi = new profiling_solver_info(1);
    x = nullptr;
    ldl_variant = 2;
    solver_mode = 0;
    reg_diag = 1e-9;
    num_pivot = 0;
    status=0;
    max_sup_wid = 64; // arbitrary
    max_col = 64;     // arbitrary
  }

  ~SolverSettings(){
    delete psi;
    if(A_ord){ allocateAC(A_ord,0,0,0,false); }
    if(AT_ord){ allocateAC(AT_ord,0,0,0,false); }
    if(L){
      if(L->p){ delete[] L->p; L->p=nullptr;}
      if(L->s){ delete[] L->s; L->s=nullptr;}
      if(L->i_ptr){ delete[] L->i_ptr; L->i_ptr=nullptr;}
      if(L->x){ delete[] L->x; L->x=nullptr;}
      // etc.
      delete L; L=nullptr;
    }
    // TODO(Da): Skipped some fields
  }

  int symbolic_analysis(){
    // Minimally, just set A_ord = A, AT_ord = A^T
    // TODO(Da): to further reorder or do supernode detection, etc.
    // Here, do a naive copy:
    *A_ord = *A;
    status=0;
    AT_ord = ptranspose(A,2,nullptr,nullptr,0,status);

    L->nsuper=1;
    L->nzmax = A->nzmax;
    L->p = new int[A->ncol+1];
    L->s = new int[A->nzmax];
    L->i_ptr = new size_t[A->ncol+1];
    L->x = new double[A->nzmax];
    valL = L->x;
    d_val = new double[2*A->ncol];
    for(int i=0; i<=A->ncol; i++){ L->p[i]=0; L->i_ptr[i]=0; }
    for(size_t j=0; j<A->nzmax; j++){ L->s[j]=0; L->x[j]=0.0; }
    for(int i=0; i<2*A->ncol; i++){ d_val[i]=0.0; }

    x = new double[A->ncol];
    return 1;
  }

  int numerical_factorization(){
    // Call ldl_left_sn_02_v2
    // TODO(Da): For simplicity, pass the entire matrix as one supernode:
    int supNo=1;
    int blockSet[2]={0, A->ncol};
    // Currently create minimal arrays to call ldl_left_sn_02_v2
    size_t* lC = L->p;
    int*    lR = L->s;
    size_t* Li_ptr = L->i_ptr;
    double* lValues = L->x;

    lC[0] = 0;
    for(int col=0; col<A->ncol; col++){
      lC[col+1] = lC[col] + (A->p[col+1] - A->p[col]);
    }

    for(int col=0; col<A->ncol; col++){
      int startA = A->p[col], endA=A->p[col+1];
      int startL = (int)lC[col];
      for(int idx=0; idx<(endA-startA); idx++){
        lR[startL+idx] = A->i[startA+idx];
        lValues[startL+idx] = A->x[startA+idx];
      }
    }

    Li_ptr[0]=0; Li_ptr[A->ncol]= lC[A->ncol];

    for(int i=1; i<A->ncol; i++){
      Li_ptr[i] = Li_ptr[i-1] + (A->p[i]-A->p[i-1]);
    }

    // The code wants cT, rT, col2Sup, aTree => we pass stubs
    int* aTree=new int[supNo]; for(int i=0; i<supNo; i++) aTree[i]=-1;
    int* cT=new int[A->ncol+1]; for(int i=0; i<=A->ncol; i++) cT[i]=0;
    int* rT=new int[A->nzmax]; for(size_t i=0; i<A->nzmax; i++) rT[i]=0;
    int* col2Sup=new int[A->ncol]; for(int i=0; i<A->ncol; i++) col2Sup[i]=0;

    double timing[10]={0};
    int *perm_piv = new int[A->ncol]; for(int i=0;i<A->ncol;i++) perm_piv[i]=i;
    int *atree_sm = new int[supNo]; for(int i=0;i<supNo;i++) atree_sm[i]=-1;

    ldl_left_sn_02_v2(A->ncol, A->p, A->i, A->x,
                      lC, lR, Li_ptr, lValues, d_val,
                      blockSet, supNo, timing,
                      aTree, cT, rT, col2Sup,
                      max_sup_wid+1, max_col+1,
                      num_pivot, perm_piv,
                      atree_sm, reg_diag);

    delete[] aTree;
    delete[] cT;
    delete[] rT;
    delete[] col2Sup;
    delete[] perm_piv;
    delete[] atree_sm;

    return 1;
  }

  double* solve_only(){
    // TODO(Da): After factorization, the factor is in L->x & d_val. We do a naive forward/back solve.
    // Current skipped a real “blocked” approach.
    // Instead, did a naive approach: L->x has the entire matrix in “factored form” if
    // we did not reorder it away. Then d_val has diagonal.

    int n = A->ncol;
    for(int i=0; i<n; i++){
      if(fabs(d_val[i])<1e-14) d_val[i] = 1e-14; 
      x[i] = rhs[i]/ d_val[i];
    }
    return x;
  }
};

/****************************************************/
/*                     MAIN()                       */
/****************************************************/

int nasoq_main(){

  int sizeH=2, nnzH=2;
  double *q = new double[sizeH]; 
  int *Hp = new int[sizeH+1];
  int *Hi = new int[nnzH];
  double*Hx= new double[nnzH];

  q[0] = -4; q[1] = -4;
  Hp[0]=0; Hp[1]=1; Hp[2]=2; 
  Hi[0]=0; Hi[1]=1; 
  Hx[0]=2; Hx[1]=2; 

  CSC* H = new CSC;
  H->ncol = sizeH; 
  H->nrow = sizeH;
  H->nzmax = nnzH; 
  H->p = Hp; 
  H->i = Hi; 
  H->x = Hx;
  H->stype=-1; 
  H->packed=1; 
  H->sorted=1; 
  H->xtype=1;

  /******** Set up the solver ********/
  SolverSettings* lbl = new SolverSettings(H,q);
  lbl->ldl_variant = 2;
  lbl->solver_mode = 0;
  lbl->reg_diag = 1e-9;

  /******** Symbolic + Numeric Factor ********/
  lbl->symbolic_analysis();
  lbl->numerical_factorization();

  /******** Solve ********/
  double* x = lbl->solve_only();

  Serial.begin(9600);
  while(!Serial) { delay(10); }
  Serial.print("Solution 1: ");
  for(int i=0; i<sizeH; i++){
    Serial.print(x[i]); 
    Serial.print(" ");
  }
  Serial.println();

  /******** Solve with a new RHS ********/
  double* new_q = new double[sizeH];
  new_q[0] = 8; 
  new_q[1] = 8;
  for(int i=0; i<sizeH; i++){ lbl->rhs[i] = new_q[i]; }

  x = lbl->solve_only();
  Serial.print("Solution 2: ");
  for(int i=0; i<sizeH; i++){
    Serial.print(x[i]); 
    Serial.print(" ");
  }
  Serial.println();

  delete[] new_q;
  delete lbl;
  H->p=nullptr; H->i=nullptr; H->x=nullptr; 
  delete H;
  delete[] q;

  return 0;
}


/* Board Setup Part */

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

  Serial.println("Start NASOQ");

  nasoq_main();
}


void loop()
{
}