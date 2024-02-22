/*
 * This file was autogenerated by OSQP on Wed Feb 21 19:38:50 2024
 * 
 * This file contains a sample solver to run the embedded code.
 */

#include <stdio.h>
#include "osqp.h"
#include "osqp_data_workspace.h"

int main() {
  OSQPInt exitflag;

  printf( "Embedded test program for vector updates.\n");

  exitflag = osqp_solve( &osqp_data_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored: %s\n", osqp_error_message(exitflag));
    return (int)exitflag;
  } else {
    printf( "  Solved workspace with no error.\n" );
  }
}
