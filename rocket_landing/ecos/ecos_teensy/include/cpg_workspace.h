
/*
Auto-generated by CVXPYgen on February 25, 2024 at 15:12:55.
Content: Type definitions and variable declarations.
*/

#include "ecos.h"

#ifndef CPG_TYPES_H
# define CPG_TYPES_H

typedef double cpg_float;
typedef int cpg_int;

// Compressed sparse column matrix
typedef struct {
  cpg_int      *p;
  cpg_int      *i;
  cpg_float    *x;
} cpg_csc;

// Canonical parameters
typedef struct {
  cpg_float    *c;         // Canonical parameter c
  cpg_float    d;          // Canonical parameter d
  cpg_csc        *A;         // Canonical parameter A
  cpg_float    *b;         // Canonical parameter b
  cpg_csc        *G;         // Canonical parameter G
  cpg_float    *h;         // Canonical parameter h
} Canon_Params_t;

// Flags indicating outdated canonical parameters
typedef struct {
  int        c;          // Bool, if canonical parameter c outdated
  int        d;          // Bool, if canonical parameter d outdated
  int        A;          // Bool, if canonical parameter A outdated
  int        b;          // Bool, if canonical parameter b outdated
  int        G;          // Bool, if canonical parameter G outdated
  int        h;          // Bool, if canonical parameter h outdated
} Canon_Outdated_t;

// Primal solution
typedef struct {
  cpg_float    *var2;      // Your variable var2
} CPG_Prim_t;

// Dual solution
typedef struct {
  cpg_float    *d0;        // Your dual variable for constraint d0
  cpg_float    *d1;        // Your dual variable for constraint d1
  cpg_float    *d2;        // Your dual variable for constraint d2
  cpg_float    *d3;        // Your dual variable for constraint d3
  cpg_float    *d4;        // Your dual variable for constraint d4
  cpg_float    *d5;        // Your dual variable for constraint d5
  cpg_float    *d6;        // Your dual variable for constraint d6
  cpg_float    *d7;        // Your dual variable for constraint d7
  cpg_float    *d8;        // Your dual variable for constraint d8
  cpg_float    *d9;        // Your dual variable for constraint d9
  cpg_float    *d10;       // Your dual variable for constraint d10
  cpg_float    *d11;       // Your dual variable for constraint d11
  cpg_float    *d12;       // Your dual variable for constraint d12
  cpg_float    *d13;       // Your dual variable for constraint d13
  cpg_float    *d14;       // Your dual variable for constraint d14
  cpg_float    *d15;       // Your dual variable for constraint d15
  cpg_float    *d16;       // Your dual variable for constraint d16
  cpg_float    *d17;       // Your dual variable for constraint d17
  cpg_float    *d18;       // Your dual variable for constraint d18
  cpg_float    *d19;       // Your dual variable for constraint d19
  cpg_float    d20;        // Your dual variable for constraint d20
  cpg_float    d21;        // Your dual variable for constraint d21
  cpg_float    d22;        // Your dual variable for constraint d22
  cpg_float    d23;        // Your dual variable for constraint d23
  cpg_float    d24;        // Your dual variable for constraint d24
  cpg_float    d25;        // Your dual variable for constraint d25
  cpg_float    d26;        // Your dual variable for constraint d26
  cpg_float    d27;        // Your dual variable for constraint d27
  cpg_float    d28;        // Your dual variable for constraint d28
  cpg_float    d29;        // Your dual variable for constraint d29
  cpg_float    d30;        // Your dual variable for constraint d30
  cpg_float    d31;        // Your dual variable for constraint d31
  cpg_float    d32;        // Your dual variable for constraint d32
  cpg_float    d33;        // Your dual variable for constraint d33
  cpg_float    d34;        // Your dual variable for constraint d34
  cpg_float    d35;        // Your dual variable for constraint d35
  cpg_float    d36;        // Your dual variable for constraint d36
  cpg_float    d37;        // Your dual variable for constraint d37
  cpg_float    d38;        // Your dual variable for constraint d38
  cpg_float    d39;        // Your dual variable for constraint d39
  cpg_float    *d40;       // Your dual variable for constraint d40
  cpg_float    *d41;       // Your dual variable for constraint d41
  cpg_float    *d42;       // Your dual variable for constraint d42
  cpg_float    *d43;       // Your dual variable for constraint d43
  cpg_float    *d44;       // Your dual variable for constraint d44
  cpg_float    *d45;       // Your dual variable for constraint d45
  cpg_float    *d46;       // Your dual variable for constraint d46
  cpg_float    *d47;       // Your dual variable for constraint d47
  cpg_float    *d48;       // Your dual variable for constraint d48
  cpg_float    *d49;       // Your dual variable for constraint d49
  cpg_float    *d50;       // Your dual variable for constraint d50
  cpg_float    *d51;       // Your dual variable for constraint d51
  cpg_float    *d52;       // Your dual variable for constraint d52
  cpg_float    *d53;       // Your dual variable for constraint d53
  cpg_float    *d54;       // Your dual variable for constraint d54
  cpg_float    *d55;       // Your dual variable for constraint d55
  cpg_float    *d56;       // Your dual variable for constraint d56
  cpg_float    *d57;       // Your dual variable for constraint d57
  cpg_float    *d58;       // Your dual variable for constraint d58
  cpg_float    *d59;       // Your dual variable for constraint d59
  cpg_float    *d60;       // Your dual variable for constraint d60
  cpg_float    *d61;       // Your dual variable for constraint d61
  cpg_float    *d62;       // Your dual variable for constraint d62
  cpg_float    *d63;       // Your dual variable for constraint d63
  cpg_float    *d64;       // Your dual variable for constraint d64
  cpg_float    *d65;       // Your dual variable for constraint d65
  cpg_float    *d66;       // Your dual variable for constraint d66
  cpg_float    *d67;       // Your dual variable for constraint d67
  cpg_float    *d68;       // Your dual variable for constraint d68
  cpg_float    *d69;       // Your dual variable for constraint d69
  cpg_float    *d70;       // Your dual variable for constraint d70
  cpg_float    *d71;       // Your dual variable for constraint d71
  cpg_float    *d72;       // Your dual variable for constraint d72
  cpg_float    *d73;       // Your dual variable for constraint d73
  cpg_float    *d74;       // Your dual variable for constraint d74
  cpg_float    *d75;       // Your dual variable for constraint d75
  cpg_float    *d76;       // Your dual variable for constraint d76
  cpg_float    *d77;       // Your dual variable for constraint d77
  cpg_float    *d78;       // Your dual variable for constraint d78
  cpg_float    *d79;       // Your dual variable for constraint d79
} CPG_Dual_t;

// Solver information
typedef struct {
  cpg_float    obj_val;    // Objective function value
  cpg_int      iter;       // Number of iterations
  cpg_int      status;     // Solver status
  cpg_float    pri_res;    // Primal residual
  cpg_float    dua_res;    // Dual residual
} CPG_Info_t;

// Solution and solver information
typedef struct {
  CPG_Prim_t *prim;      // Primal solution
  CPG_Dual_t *dual;      // Dual solution
  CPG_Info_t *info;      // Solver info
} CPG_Result_t;

// Solver settings
typedef struct {
  cpg_float  feastol;
  cpg_float  abstol;
  cpg_float  reltol;
  cpg_float  feastol_inacc;
  cpg_float  abstol_inacc;
  cpg_float  reltol_inacc;
  cpg_int    maxit;
} Canon_Settings_t;

#endif // ifndef CPG_TYPES_H

// Vector containing flattened user-defined parameters
extern cpg_float cpg_params_vec[193];

// Sparse mappings from user-defined to canonical parameters
extern cpg_csc canon_c_map;
extern cpg_csc canon_b_map;

// Canonical parameters
extern cpg_float canon_c[207];
extern cpg_float canon_c_conditioning[207];
extern cpg_csc canon_A;
extern cpg_csc canon_A_conditioning;
extern cpg_float canon_b[120];
extern cpg_float canon_b_conditioning[120];
extern cpg_csc canon_G;
extern cpg_csc canon_G_conditioning;
extern cpg_float canon_h[388];
extern cpg_float canon_h_conditioning[388];

// Struct containing canonical parameters
extern Canon_Params_t Canon_Params;
extern Canon_Params_t Canon_Params_conditioning;

// Struct containing flags for outdated canonical parameters
extern Canon_Outdated_t Canon_Outdated;

// User-defined variables
extern cpg_float cpg_var2[186];

// Dual variables associated with user-defined constraints
extern cpg_float cpg_d0[6];
extern cpg_float cpg_d1[6];
extern cpg_float cpg_d2[6];
extern cpg_float cpg_d3[6];
extern cpg_float cpg_d4[6];
extern cpg_float cpg_d5[6];
extern cpg_float cpg_d6[6];
extern cpg_float cpg_d7[6];
extern cpg_float cpg_d8[6];
extern cpg_float cpg_d9[6];
extern cpg_float cpg_d10[6];
extern cpg_float cpg_d11[6];
extern cpg_float cpg_d12[6];
extern cpg_float cpg_d13[6];
extern cpg_float cpg_d14[6];
extern cpg_float cpg_d15[6];
extern cpg_float cpg_d16[6];
extern cpg_float cpg_d17[6];
extern cpg_float cpg_d18[6];
extern cpg_float cpg_d19[6];
extern cpg_float cpg_d40[3];
extern cpg_float cpg_d41[3];
extern cpg_float cpg_d42[3];
extern cpg_float cpg_d43[3];
extern cpg_float cpg_d44[3];
extern cpg_float cpg_d45[3];
extern cpg_float cpg_d46[3];
extern cpg_float cpg_d47[3];
extern cpg_float cpg_d48[3];
extern cpg_float cpg_d49[3];
extern cpg_float cpg_d50[3];
extern cpg_float cpg_d51[3];
extern cpg_float cpg_d52[3];
extern cpg_float cpg_d53[3];
extern cpg_float cpg_d54[3];
extern cpg_float cpg_d55[3];
extern cpg_float cpg_d56[3];
extern cpg_float cpg_d57[3];
extern cpg_float cpg_d58[3];
extern cpg_float cpg_d59[3];
extern cpg_float cpg_d60[3];
extern cpg_float cpg_d61[3];
extern cpg_float cpg_d62[3];
extern cpg_float cpg_d63[3];
extern cpg_float cpg_d64[3];
extern cpg_float cpg_d65[3];
extern cpg_float cpg_d66[3];
extern cpg_float cpg_d67[3];
extern cpg_float cpg_d68[3];
extern cpg_float cpg_d69[3];
extern cpg_float cpg_d70[3];
extern cpg_float cpg_d71[3];
extern cpg_float cpg_d72[3];
extern cpg_float cpg_d73[3];
extern cpg_float cpg_d74[3];
extern cpg_float cpg_d75[3];
extern cpg_float cpg_d76[3];
extern cpg_float cpg_d77[3];
extern cpg_float cpg_d78[3];
extern cpg_float cpg_d79[3];

// Struct containing primal solution
extern CPG_Prim_t CPG_Prim;

// Struct containing dual solution
extern CPG_Dual_t CPG_Dual;

// Struct containing solver info
extern CPG_Info_t CPG_Info;

// Struct containing solution and info
extern CPG_Result_t CPG_Result;

// Struct containing solver settings
extern Canon_Settings_t Canon_Settings;

// ECOS array of SOC dimensions
extern cpg_int ecos_q[21];

// ECOS workspace
extern pwork* ecos_workspace;

// ECOS exit flag
extern cpg_int ecos_flag;
