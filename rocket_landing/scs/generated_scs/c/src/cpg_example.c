
/*
Auto-generated by CVXPYgen on February 16, 2024 at 16:59:26.
Content: Example program for updating parameters, solving, and inspecting the result.
*/
// #include "cpp_compat.h"
#include <stdio.h>
#include <stdlib.h>
#include "cpg_workspace.h"
#include "cpg_solve.h"
#include "cpg_problem.h"
#include "math.h"
#include "string.h" 

static int i;

void add_noise(float x[], float var)
{
  for (int i = 0; i < NSTATES; ++i)
  {
    float noise = ((rand() / (double)RAND_MAX) - 0.5) * 2; // random -1 to 1
    x[i] += noise * var;
  }
}

void print_vector(float xn[], int n)
{
  for (int i = 0; i < n; ++i)
  {
    // Serial.println(xn[i]);
    printf("%f, ", xn[i]);
  }
  printf("\n");
}

void matrix_vector_mult(int n1,
                        int n2,
                        float matrix[],
                        float vector[],
                        float result_vector[])
{
  // n1 is rows of matrix
  // n2 is cols of matrix, or vector
  int i, j; // i = row; j = column;
  for (i = 0; i < n1; i++)
  {
    for (j = 0; j < n2; j++)
    {
      result_vector[i] += matrix[i * n2 + j] * vector[j];
    }
  }
}

void matrix_vector_reset_mult(int n1,
                              int n2,
                              float matrix[],
                              float vector[],
                              float result_vector[])
{
  // n1 is rows of matrix
  // n2 is cols of matrix, or vector
  int i, j; // i = row; j = column;
  for (i = 0; i < n1; i++)
  {
    result_vector[i] = 0.0;
    for (j = 0; j < n2; j++)
    {
      result_vector[i] += matrix[i * n2 + j] * vector[j];
    }
  }
}

void system_dynamics(float xn[], float x[], float u[], float A[], float B[], float f[])
{
  matrix_vector_reset_mult(NSTATES, NSTATES, A, x, xn);
  matrix_vector_mult(NSTATES, NINPUTS, B, u, xn);
  for (int i = 0; i < NSTATES; ++i)
  {
    xn[i] += f[i];
  }
}

float compute_norm(float x[], float x_bar[])
{
  float res = 0.0f;
  for (int i = 0; i < NSTATES; ++i)
  {
    res += (x[i] - x_bar[i]) * (x[i] - x_bar[i]);
  }
  return sqrt(res);
}

const float xref0[] = {4, 2, 20, -3, 2, -4.5};

float xn[NSTATES] = {0};
float x[NSTATES] = {4.4, 2.2, 22, -3.3, 2.2, -4.95};
float u[NINPUTS] = {0};
float temp = 0;

int main(int argc, char *argv[]){
  // delay for 4 seconds
  // delay(500);
  // printf("Start SCS Rocket Landing\n");
  // printf("========================\n");

  // for scs
  cpg_set_solver_eps_abs(1e-2);
  cpg_set_solver_eps_rel(1e-3);
  cpg_set_solver_max_iters(500);

  srand(1);
  for (int k = 0; k < NTOTAL; ++k) {
    //// Update current measurement
    for (int i = 0; i < NSTATES; ++i)
    {
      cpg_update_param1(i, x[i]);
    }
    // printf("x = ");
    // print_vector(x, NSTATES);

    //// Update references
    for (int i = 0; i < NHORIZON; ++i)
    {
      for (int j = 0; j < NSTATES; ++j)
      {
        if (k+i >= NTOTAL)
        {
          temp = 0.0;
        }
        else
        {
          temp = xref0[j] + (0-xref0[j]) * (float)(k+i) / (NTOTAL);
        }
        // printf("temp = %f\n", temp);
        cpg_update_param3(i*(NSTATES+NINPUTS) + j, -Q_single * temp);
      }
    }
    
    // printf("%d\n", start);
    cpg_solve();
    if (k == 0) 
    {
      printf("First solve\n");
    }

    // Get data from the result
    for (i=0; i<NINPUTS; i++) {
      u[i] = CPG_Result.prim->var2[i+NSTATES];
    }
    // printf("u = ");
    // print_vector(u, NINPUTS);

    // Simulate the system
    system_dynamics(xn, x, u, A, B, f);
    // printf("xn = ");
    // print_vector(xn, NSTATES);

    // Update the state
    memcpy(x, xn, NSTATES * (sizeof(float)));
    add_noise(x, 0.01);

  }
  return 0;
}
