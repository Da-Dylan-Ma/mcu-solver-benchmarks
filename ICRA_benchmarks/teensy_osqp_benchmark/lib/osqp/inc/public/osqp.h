#ifndef OSQP_H
#define OSQP_H


/* Types, functions etc required by the OSQP API */
# include "osqp_configure.h"
# include "osqp_api_constants.h"
# include "osqp_api_types.h"
# include "osqp_api_functions.h"

#ifndef OSQP_EMBEDDED_MODE
# include "osqp_api_utils.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

const OSQPCscMatrix* osqp_get_matrix_P(const OSQPSolver* solver);
const OSQPCscMatrix* osqp_get_matrix_A(const OSQPSolver* solver);

#ifdef __cplusplus
}
#endif

#endif /* ifndef OSQP_H */
