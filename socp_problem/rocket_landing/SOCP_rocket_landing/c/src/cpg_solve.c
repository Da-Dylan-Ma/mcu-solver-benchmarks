
/*
Auto-generated by CVXPYgen on February 16, 2024 at 18:49:07.
Content: Function definitions.
*/

#include "cpg_solve.h"
#include "cpg_workspace.h"

static cpg_int i;
static cpg_int j;

// Update user-defined parameters
void cpg_update_param4(cpg_int idx, cpg_float val){
  cpg_params_vec[idx+0] = val;
  Canon_Outdated.c = 1;
}

void cpg_update_param3(cpg_int idx, cpg_float val){
  cpg_params_vec[idx+186] = val;
  Canon_Outdated.G = 1;
}

void cpg_update_param1(cpg_int idx, cpg_float val){
  cpg_params_vec[idx+34782] = val;
  Canon_Outdated.b = 1;
}

// Map user-defined to canonical parameters
void cpg_canonicalize_c(){
  for(i=0; i<207; i++){
    Canon_Params.c[i] = 0;
    for(j=canon_c_map.p[i]; j<canon_c_map.p[i+1]; j++){
      Canon_Params.c[i] += canon_c_map.x[j]*cpg_params_vec[canon_c_map.i[j]];
    }
  }
}

void cpg_canonicalize_b(){
  for(i=0; i<120; i++){
    Canon_Params.b[i] = 0;
    for(j=canon_b_map.p[i]; j<canon_b_map.p[i+1]; j++){
      Canon_Params.b[i] += canon_b_map.x[j]*cpg_params_vec[canon_b_map.i[j]];
    }
  }
}

void cpg_canonicalize_G(){
  for(i=0; i<34818; i++){
    Canon_Params.G->x[i] = 0;
    for(j=canon_G_map.p[i]; j<canon_G_map.p[i+1]; j++){
      Canon_Params.G->x[i] += canon_G_map.x[j]*cpg_params_vec[canon_G_map.i[j]];
    }
  }
}

// Retrieve primal solution in terms of user-defined variables
void cpg_retrieve_prim(){
  CPG_Prim.var2[0] = ecos_workspace->x[1];
  CPG_Prim.var2[1] = ecos_workspace->x[2];
  CPG_Prim.var2[2] = ecos_workspace->x[3];
  CPG_Prim.var2[3] = ecos_workspace->x[4];
  CPG_Prim.var2[4] = ecos_workspace->x[5];
  CPG_Prim.var2[5] = ecos_workspace->x[6];
  CPG_Prim.var2[6] = ecos_workspace->x[7];
  CPG_Prim.var2[7] = ecos_workspace->x[8];
  CPG_Prim.var2[8] = ecos_workspace->x[9];
  CPG_Prim.var2[9] = ecos_workspace->x[10];
  CPG_Prim.var2[10] = ecos_workspace->x[11];
  CPG_Prim.var2[11] = ecos_workspace->x[12];
  CPG_Prim.var2[12] = ecos_workspace->x[13];
  CPG_Prim.var2[13] = ecos_workspace->x[14];
  CPG_Prim.var2[14] = ecos_workspace->x[15];
  CPG_Prim.var2[15] = ecos_workspace->x[16];
  CPG_Prim.var2[16] = ecos_workspace->x[17];
  CPG_Prim.var2[17] = ecos_workspace->x[18];
  CPG_Prim.var2[18] = ecos_workspace->x[19];
  CPG_Prim.var2[19] = ecos_workspace->x[20];
  CPG_Prim.var2[20] = ecos_workspace->x[21];
  CPG_Prim.var2[21] = ecos_workspace->x[22];
  CPG_Prim.var2[22] = ecos_workspace->x[23];
  CPG_Prim.var2[23] = ecos_workspace->x[24];
  CPG_Prim.var2[24] = ecos_workspace->x[25];
  CPG_Prim.var2[25] = ecos_workspace->x[26];
  CPG_Prim.var2[26] = ecos_workspace->x[27];
  CPG_Prim.var2[27] = ecos_workspace->x[28];
  CPG_Prim.var2[28] = ecos_workspace->x[29];
  CPG_Prim.var2[29] = ecos_workspace->x[30];
  CPG_Prim.var2[30] = ecos_workspace->x[31];
  CPG_Prim.var2[31] = ecos_workspace->x[32];
  CPG_Prim.var2[32] = ecos_workspace->x[33];
  CPG_Prim.var2[33] = ecos_workspace->x[34];
  CPG_Prim.var2[34] = ecos_workspace->x[35];
  CPG_Prim.var2[35] = ecos_workspace->x[36];
  CPG_Prim.var2[36] = ecos_workspace->x[37];
  CPG_Prim.var2[37] = ecos_workspace->x[38];
  CPG_Prim.var2[38] = ecos_workspace->x[39];
  CPG_Prim.var2[39] = ecos_workspace->x[40];
  CPG_Prim.var2[40] = ecos_workspace->x[41];
  CPG_Prim.var2[41] = ecos_workspace->x[42];
  CPG_Prim.var2[42] = ecos_workspace->x[43];
  CPG_Prim.var2[43] = ecos_workspace->x[44];
  CPG_Prim.var2[44] = ecos_workspace->x[45];
  CPG_Prim.var2[45] = ecos_workspace->x[46];
  CPG_Prim.var2[46] = ecos_workspace->x[47];
  CPG_Prim.var2[47] = ecos_workspace->x[48];
  CPG_Prim.var2[48] = ecos_workspace->x[49];
  CPG_Prim.var2[49] = ecos_workspace->x[50];
  CPG_Prim.var2[50] = ecos_workspace->x[51];
  CPG_Prim.var2[51] = ecos_workspace->x[52];
  CPG_Prim.var2[52] = ecos_workspace->x[53];
  CPG_Prim.var2[53] = ecos_workspace->x[54];
  CPG_Prim.var2[54] = ecos_workspace->x[55];
  CPG_Prim.var2[55] = ecos_workspace->x[56];
  CPG_Prim.var2[56] = ecos_workspace->x[57];
  CPG_Prim.var2[57] = ecos_workspace->x[58];
  CPG_Prim.var2[58] = ecos_workspace->x[59];
  CPG_Prim.var2[59] = ecos_workspace->x[60];
  CPG_Prim.var2[60] = ecos_workspace->x[61];
  CPG_Prim.var2[61] = ecos_workspace->x[62];
  CPG_Prim.var2[62] = ecos_workspace->x[63];
  CPG_Prim.var2[63] = ecos_workspace->x[64];
  CPG_Prim.var2[64] = ecos_workspace->x[65];
  CPG_Prim.var2[65] = ecos_workspace->x[66];
  CPG_Prim.var2[66] = ecos_workspace->x[67];
  CPG_Prim.var2[67] = ecos_workspace->x[68];
  CPG_Prim.var2[68] = ecos_workspace->x[69];
  CPG_Prim.var2[69] = ecos_workspace->x[70];
  CPG_Prim.var2[70] = ecos_workspace->x[71];
  CPG_Prim.var2[71] = ecos_workspace->x[72];
  CPG_Prim.var2[72] = ecos_workspace->x[73];
  CPG_Prim.var2[73] = ecos_workspace->x[74];
  CPG_Prim.var2[74] = ecos_workspace->x[75];
  CPG_Prim.var2[75] = ecos_workspace->x[76];
  CPG_Prim.var2[76] = ecos_workspace->x[77];
  CPG_Prim.var2[77] = ecos_workspace->x[78];
  CPG_Prim.var2[78] = ecos_workspace->x[79];
  CPG_Prim.var2[79] = ecos_workspace->x[80];
  CPG_Prim.var2[80] = ecos_workspace->x[81];
  CPG_Prim.var2[81] = ecos_workspace->x[82];
  CPG_Prim.var2[82] = ecos_workspace->x[83];
  CPG_Prim.var2[83] = ecos_workspace->x[84];
  CPG_Prim.var2[84] = ecos_workspace->x[85];
  CPG_Prim.var2[85] = ecos_workspace->x[86];
  CPG_Prim.var2[86] = ecos_workspace->x[87];
  CPG_Prim.var2[87] = ecos_workspace->x[88];
  CPG_Prim.var2[88] = ecos_workspace->x[89];
  CPG_Prim.var2[89] = ecos_workspace->x[90];
  CPG_Prim.var2[90] = ecos_workspace->x[91];
  CPG_Prim.var2[91] = ecos_workspace->x[92];
  CPG_Prim.var2[92] = ecos_workspace->x[93];
  CPG_Prim.var2[93] = ecos_workspace->x[94];
  CPG_Prim.var2[94] = ecos_workspace->x[95];
  CPG_Prim.var2[95] = ecos_workspace->x[96];
  CPG_Prim.var2[96] = ecos_workspace->x[97];
  CPG_Prim.var2[97] = ecos_workspace->x[98];
  CPG_Prim.var2[98] = ecos_workspace->x[99];
  CPG_Prim.var2[99] = ecos_workspace->x[100];
  CPG_Prim.var2[100] = ecos_workspace->x[101];
  CPG_Prim.var2[101] = ecos_workspace->x[102];
  CPG_Prim.var2[102] = ecos_workspace->x[103];
  CPG_Prim.var2[103] = ecos_workspace->x[104];
  CPG_Prim.var2[104] = ecos_workspace->x[105];
  CPG_Prim.var2[105] = ecos_workspace->x[106];
  CPG_Prim.var2[106] = ecos_workspace->x[107];
  CPG_Prim.var2[107] = ecos_workspace->x[108];
  CPG_Prim.var2[108] = ecos_workspace->x[109];
  CPG_Prim.var2[109] = ecos_workspace->x[110];
  CPG_Prim.var2[110] = ecos_workspace->x[111];
  CPG_Prim.var2[111] = ecos_workspace->x[112];
  CPG_Prim.var2[112] = ecos_workspace->x[113];
  CPG_Prim.var2[113] = ecos_workspace->x[114];
  CPG_Prim.var2[114] = ecos_workspace->x[115];
  CPG_Prim.var2[115] = ecos_workspace->x[116];
  CPG_Prim.var2[116] = ecos_workspace->x[117];
  CPG_Prim.var2[117] = ecos_workspace->x[118];
  CPG_Prim.var2[118] = ecos_workspace->x[119];
  CPG_Prim.var2[119] = ecos_workspace->x[120];
  CPG_Prim.var2[120] = ecos_workspace->x[121];
  CPG_Prim.var2[121] = ecos_workspace->x[122];
  CPG_Prim.var2[122] = ecos_workspace->x[123];
  CPG_Prim.var2[123] = ecos_workspace->x[124];
  CPG_Prim.var2[124] = ecos_workspace->x[125];
  CPG_Prim.var2[125] = ecos_workspace->x[126];
  CPG_Prim.var2[126] = ecos_workspace->x[127];
  CPG_Prim.var2[127] = ecos_workspace->x[128];
  CPG_Prim.var2[128] = ecos_workspace->x[129];
  CPG_Prim.var2[129] = ecos_workspace->x[130];
  CPG_Prim.var2[130] = ecos_workspace->x[131];
  CPG_Prim.var2[131] = ecos_workspace->x[132];
  CPG_Prim.var2[132] = ecos_workspace->x[133];
  CPG_Prim.var2[133] = ecos_workspace->x[134];
  CPG_Prim.var2[134] = ecos_workspace->x[135];
  CPG_Prim.var2[135] = ecos_workspace->x[136];
  CPG_Prim.var2[136] = ecos_workspace->x[137];
  CPG_Prim.var2[137] = ecos_workspace->x[138];
  CPG_Prim.var2[138] = ecos_workspace->x[139];
  CPG_Prim.var2[139] = ecos_workspace->x[140];
  CPG_Prim.var2[140] = ecos_workspace->x[141];
  CPG_Prim.var2[141] = ecos_workspace->x[142];
  CPG_Prim.var2[142] = ecos_workspace->x[143];
  CPG_Prim.var2[143] = ecos_workspace->x[144];
  CPG_Prim.var2[144] = ecos_workspace->x[145];
  CPG_Prim.var2[145] = ecos_workspace->x[146];
  CPG_Prim.var2[146] = ecos_workspace->x[147];
  CPG_Prim.var2[147] = ecos_workspace->x[148];
  CPG_Prim.var2[148] = ecos_workspace->x[149];
  CPG_Prim.var2[149] = ecos_workspace->x[150];
  CPG_Prim.var2[150] = ecos_workspace->x[151];
  CPG_Prim.var2[151] = ecos_workspace->x[152];
  CPG_Prim.var2[152] = ecos_workspace->x[153];
  CPG_Prim.var2[153] = ecos_workspace->x[154];
  CPG_Prim.var2[154] = ecos_workspace->x[155];
  CPG_Prim.var2[155] = ecos_workspace->x[156];
  CPG_Prim.var2[156] = ecos_workspace->x[157];
  CPG_Prim.var2[157] = ecos_workspace->x[158];
  CPG_Prim.var2[158] = ecos_workspace->x[159];
  CPG_Prim.var2[159] = ecos_workspace->x[160];
  CPG_Prim.var2[160] = ecos_workspace->x[161];
  CPG_Prim.var2[161] = ecos_workspace->x[162];
  CPG_Prim.var2[162] = ecos_workspace->x[163];
  CPG_Prim.var2[163] = ecos_workspace->x[164];
  CPG_Prim.var2[164] = ecos_workspace->x[165];
  CPG_Prim.var2[165] = ecos_workspace->x[166];
  CPG_Prim.var2[166] = ecos_workspace->x[167];
  CPG_Prim.var2[167] = ecos_workspace->x[168];
  CPG_Prim.var2[168] = ecos_workspace->x[169];
  CPG_Prim.var2[169] = ecos_workspace->x[170];
  CPG_Prim.var2[170] = ecos_workspace->x[171];
  CPG_Prim.var2[171] = ecos_workspace->x[172];
  CPG_Prim.var2[172] = ecos_workspace->x[173];
  CPG_Prim.var2[173] = ecos_workspace->x[174];
  CPG_Prim.var2[174] = ecos_workspace->x[175];
  CPG_Prim.var2[175] = ecos_workspace->x[176];
  CPG_Prim.var2[176] = ecos_workspace->x[177];
  CPG_Prim.var2[177] = ecos_workspace->x[178];
  CPG_Prim.var2[178] = ecos_workspace->x[179];
  CPG_Prim.var2[179] = ecos_workspace->x[180];
  CPG_Prim.var2[180] = ecos_workspace->x[181];
  CPG_Prim.var2[181] = ecos_workspace->x[182];
  CPG_Prim.var2[182] = ecos_workspace->x[183];
  CPG_Prim.var2[183] = ecos_workspace->x[184];
  CPG_Prim.var2[184] = ecos_workspace->x[185];
  CPG_Prim.var2[185] = ecos_workspace->x[186];
}

// Retrieve dual solution in terms of user-defined constraints
void cpg_retrieve_dual(){
  CPG_Dual.d0[0] = ecos_workspace->y[0];
  CPG_Dual.d0[1] = ecos_workspace->y[1];
  CPG_Dual.d0[2] = ecos_workspace->y[2];
  CPG_Dual.d0[3] = ecos_workspace->y[3];
  CPG_Dual.d0[4] = ecos_workspace->y[4];
  CPG_Dual.d0[5] = ecos_workspace->y[5];
  CPG_Dual.d1[0] = ecos_workspace->y[6];
  CPG_Dual.d1[1] = ecos_workspace->y[7];
  CPG_Dual.d1[2] = ecos_workspace->y[8];
  CPG_Dual.d1[3] = ecos_workspace->y[9];
  CPG_Dual.d1[4] = ecos_workspace->y[10];
  CPG_Dual.d1[5] = ecos_workspace->y[11];
  CPG_Dual.d2[0] = ecos_workspace->y[12];
  CPG_Dual.d2[1] = ecos_workspace->y[13];
  CPG_Dual.d2[2] = ecos_workspace->y[14];
  CPG_Dual.d2[3] = ecos_workspace->y[15];
  CPG_Dual.d2[4] = ecos_workspace->y[16];
  CPG_Dual.d2[5] = ecos_workspace->y[17];
  CPG_Dual.d3[0] = ecos_workspace->y[18];
  CPG_Dual.d3[1] = ecos_workspace->y[19];
  CPG_Dual.d3[2] = ecos_workspace->y[20];
  CPG_Dual.d3[3] = ecos_workspace->y[21];
  CPG_Dual.d3[4] = ecos_workspace->y[22];
  CPG_Dual.d3[5] = ecos_workspace->y[23];
  CPG_Dual.d4[0] = ecos_workspace->y[24];
  CPG_Dual.d4[1] = ecos_workspace->y[25];
  CPG_Dual.d4[2] = ecos_workspace->y[26];
  CPG_Dual.d4[3] = ecos_workspace->y[27];
  CPG_Dual.d4[4] = ecos_workspace->y[28];
  CPG_Dual.d4[5] = ecos_workspace->y[29];
  CPG_Dual.d5[0] = ecos_workspace->y[30];
  CPG_Dual.d5[1] = ecos_workspace->y[31];
  CPG_Dual.d5[2] = ecos_workspace->y[32];
  CPG_Dual.d5[3] = ecos_workspace->y[33];
  CPG_Dual.d5[4] = ecos_workspace->y[34];
  CPG_Dual.d5[5] = ecos_workspace->y[35];
  CPG_Dual.d6[0] = ecos_workspace->y[36];
  CPG_Dual.d6[1] = ecos_workspace->y[37];
  CPG_Dual.d6[2] = ecos_workspace->y[38];
  CPG_Dual.d6[3] = ecos_workspace->y[39];
  CPG_Dual.d6[4] = ecos_workspace->y[40];
  CPG_Dual.d6[5] = ecos_workspace->y[41];
  CPG_Dual.d7[0] = ecos_workspace->y[42];
  CPG_Dual.d7[1] = ecos_workspace->y[43];
  CPG_Dual.d7[2] = ecos_workspace->y[44];
  CPG_Dual.d7[3] = ecos_workspace->y[45];
  CPG_Dual.d7[4] = ecos_workspace->y[46];
  CPG_Dual.d7[5] = ecos_workspace->y[47];
  CPG_Dual.d8[0] = ecos_workspace->y[48];
  CPG_Dual.d8[1] = ecos_workspace->y[49];
  CPG_Dual.d8[2] = ecos_workspace->y[50];
  CPG_Dual.d8[3] = ecos_workspace->y[51];
  CPG_Dual.d8[4] = ecos_workspace->y[52];
  CPG_Dual.d8[5] = ecos_workspace->y[53];
  CPG_Dual.d9[0] = ecos_workspace->y[54];
  CPG_Dual.d9[1] = ecos_workspace->y[55];
  CPG_Dual.d9[2] = ecos_workspace->y[56];
  CPG_Dual.d9[3] = ecos_workspace->y[57];
  CPG_Dual.d9[4] = ecos_workspace->y[58];
  CPG_Dual.d9[5] = ecos_workspace->y[59];
  CPG_Dual.d10[0] = ecos_workspace->y[60];
  CPG_Dual.d10[1] = ecos_workspace->y[61];
  CPG_Dual.d10[2] = ecos_workspace->y[62];
  CPG_Dual.d10[3] = ecos_workspace->y[63];
  CPG_Dual.d10[4] = ecos_workspace->y[64];
  CPG_Dual.d10[5] = ecos_workspace->y[65];
  CPG_Dual.d11[0] = ecos_workspace->y[66];
  CPG_Dual.d11[1] = ecos_workspace->y[67];
  CPG_Dual.d11[2] = ecos_workspace->y[68];
  CPG_Dual.d11[3] = ecos_workspace->y[69];
  CPG_Dual.d11[4] = ecos_workspace->y[70];
  CPG_Dual.d11[5] = ecos_workspace->y[71];
  CPG_Dual.d12[0] = ecos_workspace->y[72];
  CPG_Dual.d12[1] = ecos_workspace->y[73];
  CPG_Dual.d12[2] = ecos_workspace->y[74];
  CPG_Dual.d12[3] = ecos_workspace->y[75];
  CPG_Dual.d12[4] = ecos_workspace->y[76];
  CPG_Dual.d12[5] = ecos_workspace->y[77];
  CPG_Dual.d13[0] = ecos_workspace->y[78];
  CPG_Dual.d13[1] = ecos_workspace->y[79];
  CPG_Dual.d13[2] = ecos_workspace->y[80];
  CPG_Dual.d13[3] = ecos_workspace->y[81];
  CPG_Dual.d13[4] = ecos_workspace->y[82];
  CPG_Dual.d13[5] = ecos_workspace->y[83];
  CPG_Dual.d14[0] = ecos_workspace->y[84];
  CPG_Dual.d14[1] = ecos_workspace->y[85];
  CPG_Dual.d14[2] = ecos_workspace->y[86];
  CPG_Dual.d14[3] = ecos_workspace->y[87];
  CPG_Dual.d14[4] = ecos_workspace->y[88];
  CPG_Dual.d14[5] = ecos_workspace->y[89];
  CPG_Dual.d15[0] = ecos_workspace->y[90];
  CPG_Dual.d15[1] = ecos_workspace->y[91];
  CPG_Dual.d15[2] = ecos_workspace->y[92];
  CPG_Dual.d15[3] = ecos_workspace->y[93];
  CPG_Dual.d15[4] = ecos_workspace->y[94];
  CPG_Dual.d15[5] = ecos_workspace->y[95];
  CPG_Dual.d16[0] = ecos_workspace->y[96];
  CPG_Dual.d16[1] = ecos_workspace->y[97];
  CPG_Dual.d16[2] = ecos_workspace->y[98];
  CPG_Dual.d16[3] = ecos_workspace->y[99];
  CPG_Dual.d16[4] = ecos_workspace->y[100];
  CPG_Dual.d16[5] = ecos_workspace->y[101];
  CPG_Dual.d17[0] = ecos_workspace->y[102];
  CPG_Dual.d17[1] = ecos_workspace->y[103];
  CPG_Dual.d17[2] = ecos_workspace->y[104];
  CPG_Dual.d17[3] = ecos_workspace->y[105];
  CPG_Dual.d17[4] = ecos_workspace->y[106];
  CPG_Dual.d17[5] = ecos_workspace->y[107];
  CPG_Dual.d18[0] = ecos_workspace->y[108];
  CPG_Dual.d18[1] = ecos_workspace->y[109];
  CPG_Dual.d18[2] = ecos_workspace->y[110];
  CPG_Dual.d18[3] = ecos_workspace->y[111];
  CPG_Dual.d18[4] = ecos_workspace->y[112];
  CPG_Dual.d18[5] = ecos_workspace->y[113];
  CPG_Dual.d19[0] = ecos_workspace->y[114];
  CPG_Dual.d19[1] = ecos_workspace->y[115];
  CPG_Dual.d19[2] = ecos_workspace->y[116];
  CPG_Dual.d19[3] = ecos_workspace->y[117];
  CPG_Dual.d19[4] = ecos_workspace->y[118];
  CPG_Dual.d19[5] = ecos_workspace->y[119];
  CPG_Dual.d20 = ecos_workspace->z[0];
  CPG_Dual.d21 = ecos_workspace->z[1];
  CPG_Dual.d22 = ecos_workspace->z[2];
  CPG_Dual.d23 = ecos_workspace->z[3];
  CPG_Dual.d24 = ecos_workspace->z[4];
  CPG_Dual.d25 = ecos_workspace->z[5];
  CPG_Dual.d26 = ecos_workspace->z[6];
  CPG_Dual.d27 = ecos_workspace->z[7];
  CPG_Dual.d28 = ecos_workspace->z[8];
  CPG_Dual.d29 = ecos_workspace->z[9];
  CPG_Dual.d30 = ecos_workspace->z[10];
  CPG_Dual.d31 = ecos_workspace->z[11];
  CPG_Dual.d32 = ecos_workspace->z[12];
  CPG_Dual.d33 = ecos_workspace->z[13];
  CPG_Dual.d34 = ecos_workspace->z[14];
  CPG_Dual.d35 = ecos_workspace->z[15];
  CPG_Dual.d36 = ecos_workspace->z[16];
  CPG_Dual.d37 = ecos_workspace->z[17];
  CPG_Dual.d38 = ecos_workspace->z[18];
  CPG_Dual.d39 = ecos_workspace->z[19];
  CPG_Dual.d40[0] = ecos_workspace->z[20];
  CPG_Dual.d40[1] = ecos_workspace->z[21];
  CPG_Dual.d40[2] = ecos_workspace->z[22];
  CPG_Dual.d41[0] = ecos_workspace->z[23];
  CPG_Dual.d41[1] = ecos_workspace->z[24];
  CPG_Dual.d41[2] = ecos_workspace->z[25];
  CPG_Dual.d42[0] = ecos_workspace->z[26];
  CPG_Dual.d42[1] = ecos_workspace->z[27];
  CPG_Dual.d42[2] = ecos_workspace->z[28];
  CPG_Dual.d43[0] = ecos_workspace->z[29];
  CPG_Dual.d43[1] = ecos_workspace->z[30];
  CPG_Dual.d43[2] = ecos_workspace->z[31];
  CPG_Dual.d44[0] = ecos_workspace->z[32];
  CPG_Dual.d44[1] = ecos_workspace->z[33];
  CPG_Dual.d44[2] = ecos_workspace->z[34];
  CPG_Dual.d45[0] = ecos_workspace->z[35];
  CPG_Dual.d45[1] = ecos_workspace->z[36];
  CPG_Dual.d45[2] = ecos_workspace->z[37];
  CPG_Dual.d46[0] = ecos_workspace->z[38];
  CPG_Dual.d46[1] = ecos_workspace->z[39];
  CPG_Dual.d46[2] = ecos_workspace->z[40];
  CPG_Dual.d47[0] = ecos_workspace->z[41];
  CPG_Dual.d47[1] = ecos_workspace->z[42];
  CPG_Dual.d47[2] = ecos_workspace->z[43];
  CPG_Dual.d48[0] = ecos_workspace->z[44];
  CPG_Dual.d48[1] = ecos_workspace->z[45];
  CPG_Dual.d48[2] = ecos_workspace->z[46];
  CPG_Dual.d49[0] = ecos_workspace->z[47];
  CPG_Dual.d49[1] = ecos_workspace->z[48];
  CPG_Dual.d49[2] = ecos_workspace->z[49];
  CPG_Dual.d50[0] = ecos_workspace->z[50];
  CPG_Dual.d50[1] = ecos_workspace->z[51];
  CPG_Dual.d50[2] = ecos_workspace->z[52];
  CPG_Dual.d51[0] = ecos_workspace->z[53];
  CPG_Dual.d51[1] = ecos_workspace->z[54];
  CPG_Dual.d51[2] = ecos_workspace->z[55];
  CPG_Dual.d52[0] = ecos_workspace->z[56];
  CPG_Dual.d52[1] = ecos_workspace->z[57];
  CPG_Dual.d52[2] = ecos_workspace->z[58];
  CPG_Dual.d53[0] = ecos_workspace->z[59];
  CPG_Dual.d53[1] = ecos_workspace->z[60];
  CPG_Dual.d53[2] = ecos_workspace->z[61];
  CPG_Dual.d54[0] = ecos_workspace->z[62];
  CPG_Dual.d54[1] = ecos_workspace->z[63];
  CPG_Dual.d54[2] = ecos_workspace->z[64];
  CPG_Dual.d55[0] = ecos_workspace->z[65];
  CPG_Dual.d55[1] = ecos_workspace->z[66];
  CPG_Dual.d55[2] = ecos_workspace->z[67];
  CPG_Dual.d56[0] = ecos_workspace->z[68];
  CPG_Dual.d56[1] = ecos_workspace->z[69];
  CPG_Dual.d56[2] = ecos_workspace->z[70];
  CPG_Dual.d57[0] = ecos_workspace->z[71];
  CPG_Dual.d57[1] = ecos_workspace->z[72];
  CPG_Dual.d57[2] = ecos_workspace->z[73];
  CPG_Dual.d58[0] = ecos_workspace->z[74];
  CPG_Dual.d58[1] = ecos_workspace->z[75];
  CPG_Dual.d58[2] = ecos_workspace->z[76];
  CPG_Dual.d59[0] = ecos_workspace->z[77];
  CPG_Dual.d59[1] = ecos_workspace->z[78];
  CPG_Dual.d59[2] = ecos_workspace->z[79];
  CPG_Dual.d60[0] = ecos_workspace->z[80];
  CPG_Dual.d60[1] = ecos_workspace->z[81];
  CPG_Dual.d60[2] = ecos_workspace->z[82];
  CPG_Dual.d61[0] = ecos_workspace->z[83];
  CPG_Dual.d61[1] = ecos_workspace->z[84];
  CPG_Dual.d61[2] = ecos_workspace->z[85];
  CPG_Dual.d62[0] = ecos_workspace->z[86];
  CPG_Dual.d62[1] = ecos_workspace->z[87];
  CPG_Dual.d62[2] = ecos_workspace->z[88];
  CPG_Dual.d63[0] = ecos_workspace->z[89];
  CPG_Dual.d63[1] = ecos_workspace->z[90];
  CPG_Dual.d63[2] = ecos_workspace->z[91];
  CPG_Dual.d64[0] = ecos_workspace->z[92];
  CPG_Dual.d64[1] = ecos_workspace->z[93];
  CPG_Dual.d64[2] = ecos_workspace->z[94];
  CPG_Dual.d65[0] = ecos_workspace->z[95];
  CPG_Dual.d65[1] = ecos_workspace->z[96];
  CPG_Dual.d65[2] = ecos_workspace->z[97];
  CPG_Dual.d66[0] = ecos_workspace->z[98];
  CPG_Dual.d66[1] = ecos_workspace->z[99];
  CPG_Dual.d66[2] = ecos_workspace->z[100];
  CPG_Dual.d67[0] = ecos_workspace->z[101];
  CPG_Dual.d67[1] = ecos_workspace->z[102];
  CPG_Dual.d67[2] = ecos_workspace->z[103];
  CPG_Dual.d68[0] = ecos_workspace->z[104];
  CPG_Dual.d68[1] = ecos_workspace->z[105];
  CPG_Dual.d68[2] = ecos_workspace->z[106];
  CPG_Dual.d69[0] = ecos_workspace->z[107];
  CPG_Dual.d69[1] = ecos_workspace->z[108];
  CPG_Dual.d69[2] = ecos_workspace->z[109];
  CPG_Dual.d70[0] = ecos_workspace->z[110];
  CPG_Dual.d70[1] = ecos_workspace->z[111];
  CPG_Dual.d70[2] = ecos_workspace->z[112];
  CPG_Dual.d71[0] = ecos_workspace->z[113];
  CPG_Dual.d71[1] = ecos_workspace->z[114];
  CPG_Dual.d71[2] = ecos_workspace->z[115];
  CPG_Dual.d72[0] = ecos_workspace->z[116];
  CPG_Dual.d72[1] = ecos_workspace->z[117];
  CPG_Dual.d72[2] = ecos_workspace->z[118];
  CPG_Dual.d73[0] = ecos_workspace->z[119];
  CPG_Dual.d73[1] = ecos_workspace->z[120];
  CPG_Dual.d73[2] = ecos_workspace->z[121];
  CPG_Dual.d74[0] = ecos_workspace->z[122];
  CPG_Dual.d74[1] = ecos_workspace->z[123];
  CPG_Dual.d74[2] = ecos_workspace->z[124];
  CPG_Dual.d75[0] = ecos_workspace->z[125];
  CPG_Dual.d75[1] = ecos_workspace->z[126];
  CPG_Dual.d75[2] = ecos_workspace->z[127];
  CPG_Dual.d76[0] = ecos_workspace->z[128];
  CPG_Dual.d76[1] = ecos_workspace->z[129];
  CPG_Dual.d76[2] = ecos_workspace->z[130];
  CPG_Dual.d77[0] = ecos_workspace->z[131];
  CPG_Dual.d77[1] = ecos_workspace->z[132];
  CPG_Dual.d77[2] = ecos_workspace->z[133];
  CPG_Dual.d78[0] = ecos_workspace->z[134];
  CPG_Dual.d78[1] = ecos_workspace->z[135];
  CPG_Dual.d78[2] = ecos_workspace->z[136];
  CPG_Dual.d79[0] = ecos_workspace->z[137];
  CPG_Dual.d79[1] = ecos_workspace->z[138];
  CPG_Dual.d79[2] = ecos_workspace->z[139];
}

// Retrieve solver info
void cpg_retrieve_info(){
  CPG_Info.obj_val = (ecos_workspace->info->pcost);
  CPG_Info.iter = ecos_workspace->info->iter;
  CPG_Info.status = ecos_flag;
  CPG_Info.pri_res = ecos_workspace->info->pres;
  CPG_Info.dua_res = ecos_workspace->info->dres;
}

// Solve via canonicalization, canonical solve, retrieval
void cpg_solve(){
  // Canonicalize if necessary
  if (Canon_Outdated.c) {
    cpg_canonicalize_c();
  }
  for (i=0; i<207; i++){
    Canon_Params_conditioning.c[i] = Canon_Params.c[i];
  }
  if (Canon_Outdated.b) {
    cpg_canonicalize_b();
  }
  for (i=0; i<120; i++){
    Canon_Params_conditioning.b[i] = Canon_Params.b[i];
  }
  if (Canon_Outdated.G) {
    cpg_canonicalize_G();
  }
  for (i=0; i<34818; i++){
    Canon_Params_conditioning.G->x[i] = Canon_Params.G->x[i];
  }
  if (!ecos_workspace  ) {
    ecos_workspace = ECOS_setup(207, 388, 120, 140, 21, (int *) &ecos_q, 0, Canon_Params_conditioning.G->x, Canon_Params_conditioning.G->p, Canon_Params_conditioning.G->i, Canon_Params_conditioning.A->x, Canon_Params_conditioning.A->p, Canon_Params_conditioning.A->i, Canon_Params_conditioning.c, Canon_Params_conditioning.h, Canon_Params_conditioning.b);
  } else {
    if ( Canon_Outdated.A||Canon_Outdated.b||Canon_Outdated.G) {
      ECOS_updateData(ecos_workspace, Canon_Params_conditioning.G->x, Canon_Params_conditioning.A->x, Canon_Params_conditioning.c, Canon_Params_conditioning.h, Canon_Params_conditioning.b);
    } else {
      if ( Canon_Outdated.c) {
        for (i=0; i<207; i++) { ecos_updateDataEntry_c(ecos_workspace, i, Canon_Params_conditioning.c[i]); };
      }
    }
  }
  ecos_workspace->stgs->feastol = Canon_Settings.feastol;
  ecos_workspace->stgs->abstol = Canon_Settings.abstol;
  ecos_workspace->stgs->reltol = Canon_Settings.reltol;
  ecos_workspace->stgs->feastol_inacc = Canon_Settings.feastol_inacc;
  ecos_workspace->stgs->abstol_inacc = Canon_Settings.abstol_inacc;
  ecos_workspace->stgs->reltol_inacc = Canon_Settings.reltol_inacc;
  ecos_workspace->stgs->maxit = Canon_Settings.maxit;
  // Solve with ECOS
  ecos_flag = ECOS_solve(ecos_workspace);
  // Retrieve results
  cpg_retrieve_prim();
  cpg_retrieve_dual();
  cpg_retrieve_info();
  // Reset flags for outdated canonical parameters
  Canon_Outdated.c = 0;
  Canon_Outdated.b = 0;
  Canon_Outdated.G = 0;
}

// Update solver settings
void cpg_set_solver_default_settings(){
  Canon_Settings.feastol = 1e-8;
  Canon_Settings.abstol = 1e-8;
  Canon_Settings.reltol = 1e-8;
  Canon_Settings.feastol_inacc = 1e-4;
  Canon_Settings.abstol_inacc = 5e-5;
  Canon_Settings.reltol_inacc = 5e-5;
  Canon_Settings.maxit = 100;
}

void cpg_set_solver_feastol(cpg_float feastol_new){
  Canon_Settings.feastol = feastol_new;
}

void cpg_set_solver_abstol(cpg_float abstol_new){
  Canon_Settings.abstol = abstol_new;
}

void cpg_set_solver_reltol(cpg_float reltol_new){
  Canon_Settings.reltol = reltol_new;
}

void cpg_set_solver_feastol_inacc(cpg_float feastol_inacc_new){
  Canon_Settings.feastol_inacc = feastol_inacc_new;
}

void cpg_set_solver_abstol_inacc(cpg_float abstol_inacc_new){
  Canon_Settings.abstol_inacc = abstol_inacc_new;
}

void cpg_set_solver_reltol_inacc(cpg_float reltol_inacc_new){
  Canon_Settings.reltol_inacc = reltol_inacc_new;
}

void cpg_set_solver_maxit(cpg_int maxit_new){
  Canon_Settings.maxit = maxit_new;
}
