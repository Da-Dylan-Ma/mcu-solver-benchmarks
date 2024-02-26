
/*
Auto-generated by CVXPYgen on February 25, 2024 at 22:24:38.
Content: Python binding with pybind11.
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ctime>
#include "cpg_module.hpp"

extern "C" {
    #include "include/cpg_workspace.h"
    #include "include/cpg_solve.h"
}

namespace py = pybind11;

static int i;

CPG_Result_cpp_t solve_cpp(struct CPG_Updated_cpp_t& CPG_Updated_cpp, struct CPG_Params_cpp_t& CPG_Params_cpp){

    // Pass changed user-defined parameter values to the solver
    if (CPG_Updated_cpp.param3) {
        for(i=0; i<285; i++) {
            cpg_update_param3(i, CPG_Params_cpp.param3[i]);
        }
    }
    if (CPG_Updated_cpp.param1) {
        for(i=0; i<6; i++) {
            cpg_update_param1(i, CPG_Params_cpp.param1[i]);
        }
    }

    // Solve
    std::clock_t ASA_start = std::clock();
    cpg_solve();
    std::clock_t ASA_end = std::clock();

    // Arrange and return results
    CPG_Prim_cpp_t CPG_Prim_cpp {};
    for(i=0; i<285; i++) {
        CPG_Prim_cpp.var2[i] = CPG_Prim.var2[i];
    }
    CPG_Dual_cpp_t CPG_Dual_cpp {};
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d0[i] = CPG_Dual.d0[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d1[i] = CPG_Dual.d1[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d2[i] = CPG_Dual.d2[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d3[i] = CPG_Dual.d3[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d4[i] = CPG_Dual.d4[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d5[i] = CPG_Dual.d5[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d6[i] = CPG_Dual.d6[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d7[i] = CPG_Dual.d7[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d8[i] = CPG_Dual.d8[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d9[i] = CPG_Dual.d9[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d10[i] = CPG_Dual.d10[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d11[i] = CPG_Dual.d11[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d12[i] = CPG_Dual.d12[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d13[i] = CPG_Dual.d13[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d14[i] = CPG_Dual.d14[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d15[i] = CPG_Dual.d15[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d16[i] = CPG_Dual.d16[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d17[i] = CPG_Dual.d17[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d18[i] = CPG_Dual.d18[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d19[i] = CPG_Dual.d19[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d20[i] = CPG_Dual.d20[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d21[i] = CPG_Dual.d21[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d22[i] = CPG_Dual.d22[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d23[i] = CPG_Dual.d23[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d24[i] = CPG_Dual.d24[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d25[i] = CPG_Dual.d25[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d26[i] = CPG_Dual.d26[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d27[i] = CPG_Dual.d27[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d28[i] = CPG_Dual.d28[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d29[i] = CPG_Dual.d29[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d30[i] = CPG_Dual.d30[i];
    }
    for(i=0; i<6; i++) {
        CPG_Dual_cpp.d31[i] = CPG_Dual.d31[i];
    }
    CPG_Dual_cpp.d32 = CPG_Dual.d32;
    CPG_Dual_cpp.d33 = CPG_Dual.d33;
    CPG_Dual_cpp.d34 = CPG_Dual.d34;
    CPG_Dual_cpp.d35 = CPG_Dual.d35;
    CPG_Dual_cpp.d36 = CPG_Dual.d36;
    CPG_Dual_cpp.d37 = CPG_Dual.d37;
    CPG_Dual_cpp.d38 = CPG_Dual.d38;
    CPG_Dual_cpp.d39 = CPG_Dual.d39;
    CPG_Dual_cpp.d40 = CPG_Dual.d40;
    CPG_Dual_cpp.d41 = CPG_Dual.d41;
    CPG_Dual_cpp.d42 = CPG_Dual.d42;
    CPG_Dual_cpp.d43 = CPG_Dual.d43;
    CPG_Dual_cpp.d44 = CPG_Dual.d44;
    CPG_Dual_cpp.d45 = CPG_Dual.d45;
    CPG_Dual_cpp.d46 = CPG_Dual.d46;
    CPG_Dual_cpp.d47 = CPG_Dual.d47;
    CPG_Dual_cpp.d48 = CPG_Dual.d48;
    CPG_Dual_cpp.d49 = CPG_Dual.d49;
    CPG_Dual_cpp.d50 = CPG_Dual.d50;
    CPG_Dual_cpp.d51 = CPG_Dual.d51;
    CPG_Dual_cpp.d52 = CPG_Dual.d52;
    CPG_Dual_cpp.d53 = CPG_Dual.d53;
    CPG_Dual_cpp.d54 = CPG_Dual.d54;
    CPG_Dual_cpp.d55 = CPG_Dual.d55;
    CPG_Dual_cpp.d56 = CPG_Dual.d56;
    CPG_Dual_cpp.d57 = CPG_Dual.d57;
    CPG_Dual_cpp.d58 = CPG_Dual.d58;
    CPG_Dual_cpp.d59 = CPG_Dual.d59;
    CPG_Dual_cpp.d60 = CPG_Dual.d60;
    CPG_Dual_cpp.d61 = CPG_Dual.d61;
    CPG_Dual_cpp.d62 = CPG_Dual.d62;
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d63[i] = CPG_Dual.d63[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d64[i] = CPG_Dual.d64[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d65[i] = CPG_Dual.d65[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d66[i] = CPG_Dual.d66[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d67[i] = CPG_Dual.d67[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d68[i] = CPG_Dual.d68[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d69[i] = CPG_Dual.d69[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d70[i] = CPG_Dual.d70[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d71[i] = CPG_Dual.d71[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d72[i] = CPG_Dual.d72[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d73[i] = CPG_Dual.d73[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d74[i] = CPG_Dual.d74[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d75[i] = CPG_Dual.d75[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d76[i] = CPG_Dual.d76[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d77[i] = CPG_Dual.d77[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d78[i] = CPG_Dual.d78[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d79[i] = CPG_Dual.d79[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d80[i] = CPG_Dual.d80[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d81[i] = CPG_Dual.d81[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d82[i] = CPG_Dual.d82[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d83[i] = CPG_Dual.d83[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d84[i] = CPG_Dual.d84[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d85[i] = CPG_Dual.d85[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d86[i] = CPG_Dual.d86[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d87[i] = CPG_Dual.d87[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d88[i] = CPG_Dual.d88[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d89[i] = CPG_Dual.d89[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d90[i] = CPG_Dual.d90[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d91[i] = CPG_Dual.d91[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d92[i] = CPG_Dual.d92[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d93[i] = CPG_Dual.d93[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d94[i] = CPG_Dual.d94[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d95[i] = CPG_Dual.d95[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d96[i] = CPG_Dual.d96[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d97[i] = CPG_Dual.d97[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d98[i] = CPG_Dual.d98[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d99[i] = CPG_Dual.d99[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d100[i] = CPG_Dual.d100[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d101[i] = CPG_Dual.d101[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d102[i] = CPG_Dual.d102[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d103[i] = CPG_Dual.d103[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d104[i] = CPG_Dual.d104[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d105[i] = CPG_Dual.d105[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d106[i] = CPG_Dual.d106[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d107[i] = CPG_Dual.d107[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d108[i] = CPG_Dual.d108[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d109[i] = CPG_Dual.d109[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d110[i] = CPG_Dual.d110[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d111[i] = CPG_Dual.d111[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d112[i] = CPG_Dual.d112[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d113[i] = CPG_Dual.d113[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d114[i] = CPG_Dual.d114[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d115[i] = CPG_Dual.d115[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d116[i] = CPG_Dual.d116[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d117[i] = CPG_Dual.d117[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d118[i] = CPG_Dual.d118[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d119[i] = CPG_Dual.d119[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d120[i] = CPG_Dual.d120[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d121[i] = CPG_Dual.d121[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d122[i] = CPG_Dual.d122[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d123[i] = CPG_Dual.d123[i];
    }
    for(i=0; i<3; i++) {
        CPG_Dual_cpp.d124[i] = CPG_Dual.d124[i];
    }
    CPG_Info_cpp_t CPG_Info_cpp {};
    CPG_Info_cpp.obj_val = CPG_Info.obj_val;
    CPG_Info_cpp.iter = CPG_Info.iter;
    CPG_Info_cpp.status = CPG_Info.status;
    CPG_Info_cpp.pri_res = CPG_Info.pri_res;
    CPG_Info_cpp.dua_res = CPG_Info.dua_res;
    CPG_Info_cpp.time = 1.0 * (ASA_end - ASA_start) / CLOCKS_PER_SEC;
    CPG_Result_cpp_t CPG_Result_cpp {};
    CPG_Result_cpp.prim = CPG_Prim_cpp;
    CPG_Result_cpp.dual = CPG_Dual_cpp;
    CPG_Result_cpp.info = CPG_Info_cpp;
    return CPG_Result_cpp;

}

PYBIND11_MODULE(cpg_module, m) {

    py::class_<CPG_Params_cpp_t>(m, "cpg_params")
            .def(py::init<>())
            .def_readwrite("param3", &CPG_Params_cpp_t::param3)
            .def_readwrite("param1", &CPG_Params_cpp_t::param1)
            ;

    py::class_<CPG_Updated_cpp_t>(m, "cpg_updated")
            .def(py::init<>())
            .def_readwrite("param3", &CPG_Updated_cpp_t::param3)
            .def_readwrite("param1", &CPG_Updated_cpp_t::param1)
            ;

    py::class_<CPG_Prim_cpp_t>(m, "cpg_prim")
            .def(py::init<>())
            .def_readwrite("var2", &CPG_Prim_cpp_t::var2)
            ;

    py::class_<CPG_Dual_cpp_t>(m, "cpg_dual")
            .def(py::init<>())
            .def_readwrite("d0", &CPG_Dual_cpp_t::d0)
            .def_readwrite("d1", &CPG_Dual_cpp_t::d1)
            .def_readwrite("d2", &CPG_Dual_cpp_t::d2)
            .def_readwrite("d3", &CPG_Dual_cpp_t::d3)
            .def_readwrite("d4", &CPG_Dual_cpp_t::d4)
            .def_readwrite("d5", &CPG_Dual_cpp_t::d5)
            .def_readwrite("d6", &CPG_Dual_cpp_t::d6)
            .def_readwrite("d7", &CPG_Dual_cpp_t::d7)
            .def_readwrite("d8", &CPG_Dual_cpp_t::d8)
            .def_readwrite("d9", &CPG_Dual_cpp_t::d9)
            .def_readwrite("d10", &CPG_Dual_cpp_t::d10)
            .def_readwrite("d11", &CPG_Dual_cpp_t::d11)
            .def_readwrite("d12", &CPG_Dual_cpp_t::d12)
            .def_readwrite("d13", &CPG_Dual_cpp_t::d13)
            .def_readwrite("d14", &CPG_Dual_cpp_t::d14)
            .def_readwrite("d15", &CPG_Dual_cpp_t::d15)
            .def_readwrite("d16", &CPG_Dual_cpp_t::d16)
            .def_readwrite("d17", &CPG_Dual_cpp_t::d17)
            .def_readwrite("d18", &CPG_Dual_cpp_t::d18)
            .def_readwrite("d19", &CPG_Dual_cpp_t::d19)
            .def_readwrite("d20", &CPG_Dual_cpp_t::d20)
            .def_readwrite("d21", &CPG_Dual_cpp_t::d21)
            .def_readwrite("d22", &CPG_Dual_cpp_t::d22)
            .def_readwrite("d23", &CPG_Dual_cpp_t::d23)
            .def_readwrite("d24", &CPG_Dual_cpp_t::d24)
            .def_readwrite("d25", &CPG_Dual_cpp_t::d25)
            .def_readwrite("d26", &CPG_Dual_cpp_t::d26)
            .def_readwrite("d27", &CPG_Dual_cpp_t::d27)
            .def_readwrite("d28", &CPG_Dual_cpp_t::d28)
            .def_readwrite("d29", &CPG_Dual_cpp_t::d29)
            .def_readwrite("d30", &CPG_Dual_cpp_t::d30)
            .def_readwrite("d31", &CPG_Dual_cpp_t::d31)
            .def_readwrite("d32", &CPG_Dual_cpp_t::d32)
            .def_readwrite("d33", &CPG_Dual_cpp_t::d33)
            .def_readwrite("d34", &CPG_Dual_cpp_t::d34)
            .def_readwrite("d35", &CPG_Dual_cpp_t::d35)
            .def_readwrite("d36", &CPG_Dual_cpp_t::d36)
            .def_readwrite("d37", &CPG_Dual_cpp_t::d37)
            .def_readwrite("d38", &CPG_Dual_cpp_t::d38)
            .def_readwrite("d39", &CPG_Dual_cpp_t::d39)
            .def_readwrite("d40", &CPG_Dual_cpp_t::d40)
            .def_readwrite("d41", &CPG_Dual_cpp_t::d41)
            .def_readwrite("d42", &CPG_Dual_cpp_t::d42)
            .def_readwrite("d43", &CPG_Dual_cpp_t::d43)
            .def_readwrite("d44", &CPG_Dual_cpp_t::d44)
            .def_readwrite("d45", &CPG_Dual_cpp_t::d45)
            .def_readwrite("d46", &CPG_Dual_cpp_t::d46)
            .def_readwrite("d47", &CPG_Dual_cpp_t::d47)
            .def_readwrite("d48", &CPG_Dual_cpp_t::d48)
            .def_readwrite("d49", &CPG_Dual_cpp_t::d49)
            .def_readwrite("d50", &CPG_Dual_cpp_t::d50)
            .def_readwrite("d51", &CPG_Dual_cpp_t::d51)
            .def_readwrite("d52", &CPG_Dual_cpp_t::d52)
            .def_readwrite("d53", &CPG_Dual_cpp_t::d53)
            .def_readwrite("d54", &CPG_Dual_cpp_t::d54)
            .def_readwrite("d55", &CPG_Dual_cpp_t::d55)
            .def_readwrite("d56", &CPG_Dual_cpp_t::d56)
            .def_readwrite("d57", &CPG_Dual_cpp_t::d57)
            .def_readwrite("d58", &CPG_Dual_cpp_t::d58)
            .def_readwrite("d59", &CPG_Dual_cpp_t::d59)
            .def_readwrite("d60", &CPG_Dual_cpp_t::d60)
            .def_readwrite("d61", &CPG_Dual_cpp_t::d61)
            .def_readwrite("d62", &CPG_Dual_cpp_t::d62)
            .def_readwrite("d63", &CPG_Dual_cpp_t::d63)
            .def_readwrite("d64", &CPG_Dual_cpp_t::d64)
            .def_readwrite("d65", &CPG_Dual_cpp_t::d65)
            .def_readwrite("d66", &CPG_Dual_cpp_t::d66)
            .def_readwrite("d67", &CPG_Dual_cpp_t::d67)
            .def_readwrite("d68", &CPG_Dual_cpp_t::d68)
            .def_readwrite("d69", &CPG_Dual_cpp_t::d69)
            .def_readwrite("d70", &CPG_Dual_cpp_t::d70)
            .def_readwrite("d71", &CPG_Dual_cpp_t::d71)
            .def_readwrite("d72", &CPG_Dual_cpp_t::d72)
            .def_readwrite("d73", &CPG_Dual_cpp_t::d73)
            .def_readwrite("d74", &CPG_Dual_cpp_t::d74)
            .def_readwrite("d75", &CPG_Dual_cpp_t::d75)
            .def_readwrite("d76", &CPG_Dual_cpp_t::d76)
            .def_readwrite("d77", &CPG_Dual_cpp_t::d77)
            .def_readwrite("d78", &CPG_Dual_cpp_t::d78)
            .def_readwrite("d79", &CPG_Dual_cpp_t::d79)
            .def_readwrite("d80", &CPG_Dual_cpp_t::d80)
            .def_readwrite("d81", &CPG_Dual_cpp_t::d81)
            .def_readwrite("d82", &CPG_Dual_cpp_t::d82)
            .def_readwrite("d83", &CPG_Dual_cpp_t::d83)
            .def_readwrite("d84", &CPG_Dual_cpp_t::d84)
            .def_readwrite("d85", &CPG_Dual_cpp_t::d85)
            .def_readwrite("d86", &CPG_Dual_cpp_t::d86)
            .def_readwrite("d87", &CPG_Dual_cpp_t::d87)
            .def_readwrite("d88", &CPG_Dual_cpp_t::d88)
            .def_readwrite("d89", &CPG_Dual_cpp_t::d89)
            .def_readwrite("d90", &CPG_Dual_cpp_t::d90)
            .def_readwrite("d91", &CPG_Dual_cpp_t::d91)
            .def_readwrite("d92", &CPG_Dual_cpp_t::d92)
            .def_readwrite("d93", &CPG_Dual_cpp_t::d93)
            .def_readwrite("d94", &CPG_Dual_cpp_t::d94)
            .def_readwrite("d95", &CPG_Dual_cpp_t::d95)
            .def_readwrite("d96", &CPG_Dual_cpp_t::d96)
            .def_readwrite("d97", &CPG_Dual_cpp_t::d97)
            .def_readwrite("d98", &CPG_Dual_cpp_t::d98)
            .def_readwrite("d99", &CPG_Dual_cpp_t::d99)
            .def_readwrite("d100", &CPG_Dual_cpp_t::d100)
            .def_readwrite("d101", &CPG_Dual_cpp_t::d101)
            .def_readwrite("d102", &CPG_Dual_cpp_t::d102)
            .def_readwrite("d103", &CPG_Dual_cpp_t::d103)
            .def_readwrite("d104", &CPG_Dual_cpp_t::d104)
            .def_readwrite("d105", &CPG_Dual_cpp_t::d105)
            .def_readwrite("d106", &CPG_Dual_cpp_t::d106)
            .def_readwrite("d107", &CPG_Dual_cpp_t::d107)
            .def_readwrite("d108", &CPG_Dual_cpp_t::d108)
            .def_readwrite("d109", &CPG_Dual_cpp_t::d109)
            .def_readwrite("d110", &CPG_Dual_cpp_t::d110)
            .def_readwrite("d111", &CPG_Dual_cpp_t::d111)
            .def_readwrite("d112", &CPG_Dual_cpp_t::d112)
            .def_readwrite("d113", &CPG_Dual_cpp_t::d113)
            .def_readwrite("d114", &CPG_Dual_cpp_t::d114)
            .def_readwrite("d115", &CPG_Dual_cpp_t::d115)
            .def_readwrite("d116", &CPG_Dual_cpp_t::d116)
            .def_readwrite("d117", &CPG_Dual_cpp_t::d117)
            .def_readwrite("d118", &CPG_Dual_cpp_t::d118)
            .def_readwrite("d119", &CPG_Dual_cpp_t::d119)
            .def_readwrite("d120", &CPG_Dual_cpp_t::d120)
            .def_readwrite("d121", &CPG_Dual_cpp_t::d121)
            .def_readwrite("d122", &CPG_Dual_cpp_t::d122)
            .def_readwrite("d123", &CPG_Dual_cpp_t::d123)
            .def_readwrite("d124", &CPG_Dual_cpp_t::d124)
            ;

    py::class_<CPG_Info_cpp_t>(m, "cpg_info")
            .def(py::init<>())
            .def_readwrite("obj_val", &CPG_Info_cpp_t::obj_val)
            .def_readwrite("iter", &CPG_Info_cpp_t::iter)
            .def_readwrite("status", &CPG_Info_cpp_t::status)
            .def_readwrite("pri_res", &CPG_Info_cpp_t::pri_res)
            .def_readwrite("dua_res", &CPG_Info_cpp_t::dua_res)
            .def_readwrite("time", &CPG_Info_cpp_t::time)
            ;

    py::class_<CPG_Result_cpp_t>(m, "cpg_result")
            .def(py::init<>())
            .def_readwrite("cpg_prim", &CPG_Result_cpp_t::prim)
            .def_readwrite("cpg_dual", &CPG_Result_cpp_t::dual)
            .def_readwrite("cpg_info", &CPG_Result_cpp_t::info)
            ;

    m.def("solve", &solve_cpp);

    m.def("set_solver_default_settings", &cpg_set_solver_default_settings);
    m.def("set_solver_normalize", &cpg_set_solver_normalize);
    m.def("set_solver_scale", &cpg_set_solver_scale);
    m.def("set_solver_adaptive_scale", &cpg_set_solver_adaptive_scale);
    m.def("set_solver_rho_x", &cpg_set_solver_rho_x);
    m.def("set_solver_max_iters", &cpg_set_solver_max_iters);
    m.def("set_solver_eps_abs", &cpg_set_solver_eps_abs);
    m.def("set_solver_eps_rel", &cpg_set_solver_eps_rel);
    m.def("set_solver_eps_infeas", &cpg_set_solver_eps_infeas);
    m.def("set_solver_alpha", &cpg_set_solver_alpha);
    m.def("set_solver_time_limit_secs", &cpg_set_solver_time_limit_secs);
    m.def("set_solver_verbose", &cpg_set_solver_verbose);
    m.def("set_solver_warm_start", &cpg_set_solver_warm_start);
    m.def("set_solver_acceleration_lookback", &cpg_set_solver_acceleration_lookback);
    m.def("set_solver_acceleration_interval", &cpg_set_solver_acceleration_interval);
    m.def("set_solver_write_data_filename", &cpg_set_solver_write_data_filename);
    m.def("set_solver_log_csv_filename", &cpg_set_solver_log_csv_filename);

}
