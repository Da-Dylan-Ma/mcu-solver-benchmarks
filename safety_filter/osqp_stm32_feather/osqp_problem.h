#include "src/osqp/inc/public/osqp_api_types.h"

#pragma once

#define NSTATES 10

#define NINPUTS 5

#define NHORIZON 20

#define NTOTAL 201

#define SIZE_Q 295

#define SIZE_LU 495

const PROGMEM OSQPFloat mR[10*10] = {
  -100.0,	-0.0,	-0.0,	-0.0,	-0.0,	
  -0.0,	-100.0,	-0.0,	-0.0,	-0.0,	
  -0.0,	-0.0,	-100.0,	-0.0,	-0.0,	
  -0.0,	-0.0,	-0.0,	-100.0,	-0.0,	
  -0.0,	-0.0,	-0.0,	-0.0,	-100.0,	
};

const PROGMEM OSQPFloat A[10*10] = {
  1.0,	0.0,	0.0,	0.0,	0.0,	0.05,	0.0,	0.0,	0.0,	0.0,	
  0.0,	1.0,	0.0,	0.0,	0.0,	0.0,	0.05,	0.0,	0.0,	0.0,	
  0.0,	0.0,	1.0,	0.0,	0.0,	0.0,	0.0,	0.05,	0.0,	0.0,	
  0.0,	0.0,	0.0,	1.0,	0.0,	0.0,	0.0,	0.0,	0.05,	0.0,	
  0.0,	0.0,	0.0,	0.0,	1.0,	0.0,	0.0,	0.0,	0.0,	0.05,	
  0.0,	0.0,	0.0,	0.0,	0.0,	1.0,	0.0,	0.0,	0.0,	0.0,	
  0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	1.0,	0.0,	0.0,	0.0,	
  0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	1.0,	0.0,	0.0,	
  0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	1.0,	0.0,	
  0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	1.0,	
};

const PROGMEM OSQPFloat B[10*5] = {
  0.0012500000000000002,	0.0,	0.0,	0.0,	0.0,	
  0.0,	0.0012500000000000002,	0.0,	0.0,	0.0,	
  0.0,	0.0,	0.0012500000000000002,	0.0,	0.0,	
  0.0,	0.0,	0.0,	0.0012500000000000002,	0.0,	
  0.0,	0.0,	0.0,	0.0,	0.0012500000000000002,	
  0.05,	0.0,	0.0,	0.0,	0.0,	
  0.0,	0.05,	0.0,	0.0,	0.0,	
  0.0,	0.0,	0.05,	0.0,	0.0,	
  0.0,	0.0,	0.0,	0.05,	0.0,	
  0.0,	0.0,	0.0,	0.0,	0.05,	
};

