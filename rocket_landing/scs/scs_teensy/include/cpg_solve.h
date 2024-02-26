
/*
Auto-generated by CVXPYgen on February 25, 2024 at 22:24:38.
Content: Function declarations.
*/

#include "cpg_workspace.h"

// Update user-defined parameter values
extern void cpg_update_param3(cpg_int idx, cpg_float val);
extern void cpg_update_param1(cpg_int idx, cpg_float val);

// Map user-defined to canonical parameters
extern void cpg_canonicalize_c();
extern void cpg_canonicalize_b();

// Retrieve dual solution in terms of user-defined constraints
extern void cpg_retrieve_dual();

// Retrieve solver information
extern void cpg_retrieve_info();

// Solve via canonicalization, canonical solve, retrieval
extern void cpg_solve();

// Update solver settings
extern void cpg_set_solver_default_settings();
extern void cpg_set_solver_normalize(cpg_int normalize_new);
extern void cpg_set_solver_scale(cpg_float scale_new);
extern void cpg_set_solver_adaptive_scale(cpg_int adaptive_scale_new);
extern void cpg_set_solver_rho_x(cpg_float rho_x_new);
extern void cpg_set_solver_max_iters(cpg_int max_iters_new);
extern void cpg_set_solver_eps_abs(cpg_float eps_abs_new);
extern void cpg_set_solver_eps_rel(cpg_float eps_rel_new);
extern void cpg_set_solver_eps_infeas(cpg_float eps_infeas_new);
extern void cpg_set_solver_alpha(cpg_float alpha_new);
extern void cpg_set_solver_time_limit_secs(cpg_float time_limit_secs_new);
extern void cpg_set_solver_verbose(cpg_int verbose_new);
extern void cpg_set_solver_warm_start(cpg_int warm_start_new);
extern void cpg_set_solver_acceleration_lookback(cpg_int acceleration_lookback_new);
extern void cpg_set_solver_acceleration_interval(cpg_int acceleration_interval_new);
extern void cpg_set_solver_write_data_filename(const char* write_data_filename_new);
extern void cpg_set_solver_log_csv_filename(const char* log_csv_filename_new);
