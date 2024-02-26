
"""
Auto-generated by CVXPYgen on February 26, 2024 at 17:03:52.
Content: Custom solve method for CVXPY interface.
"""

import time
import warnings
import numpy as np
from cvxpy.reductions import Solution
from cvxpy.problems.problem import SolverStats
from scs.generated_scs import cpg_module


standard_settings_names = {"max_iters": "maxit"}


def cpg_solve(prob, updated_params=None, **kwargs):

    # set flags for updated parameters
    upd = cpg_module.cpg_updated()
    if updated_params is None:
        updated_params = ["param3", "param1"]
    for p in updated_params:
        try:
            setattr(upd, p, True)
        except AttributeError:
            raise AttributeError(f"{p} is not a parameter.")

    # set solver settings
    cpg_module.set_solver_default_settings()
    for key, value in kwargs.items():
        try:
            eval(f'cpg_module.set_solver_{standard_settings_names.get(key, key)}(value)')
        except AttributeError:
            raise AttributeError(f'Solver setting "{key}" not available.')

    # set parameter values
    par = cpg_module.cpg_params()
    param_dict = prob.param_dict
    par.param3 = list(param_dict["param3"].value.flatten(order="F"))
    par.param1 = list(param_dict["param1"].value.flatten(order="F"))

    # solve
    t0 = time.time()
    res = cpg_module.solve(upd, par)
    t1 = time.time()

    # store solution in problem object
    prob._clear_solution()
    prob.var_dict['var2'].save_value(np.array(res.cpg_prim.var2).reshape(285))
    prob.constraints[0].save_dual_value(np.array(res.cpg_dual.d0).reshape(6))
    prob.constraints[1].save_dual_value(np.array(res.cpg_dual.d1).reshape(6))
    prob.constraints[2].save_dual_value(np.array(res.cpg_dual.d2).reshape(6))
    prob.constraints[3].save_dual_value(np.array(res.cpg_dual.d3).reshape(6))
    prob.constraints[4].save_dual_value(np.array(res.cpg_dual.d4).reshape(6))
    prob.constraints[5].save_dual_value(np.array(res.cpg_dual.d5).reshape(6))
    prob.constraints[6].save_dual_value(np.array(res.cpg_dual.d6).reshape(6))
    prob.constraints[7].save_dual_value(np.array(res.cpg_dual.d7).reshape(6))
    prob.constraints[8].save_dual_value(np.array(res.cpg_dual.d8).reshape(6))
    prob.constraints[9].save_dual_value(np.array(res.cpg_dual.d9).reshape(6))
    prob.constraints[10].save_dual_value(np.array(res.cpg_dual.d10).reshape(6))
    prob.constraints[11].save_dual_value(np.array(res.cpg_dual.d11).reshape(6))
    prob.constraints[12].save_dual_value(np.array(res.cpg_dual.d12).reshape(6))
    prob.constraints[13].save_dual_value(np.array(res.cpg_dual.d13).reshape(6))
    prob.constraints[14].save_dual_value(np.array(res.cpg_dual.d14).reshape(6))
    prob.constraints[15].save_dual_value(np.array(res.cpg_dual.d15).reshape(6))
    prob.constraints[16].save_dual_value(np.array(res.cpg_dual.d16).reshape(6))
    prob.constraints[17].save_dual_value(np.array(res.cpg_dual.d17).reshape(6))
    prob.constraints[18].save_dual_value(np.array(res.cpg_dual.d18).reshape(6))
    prob.constraints[19].save_dual_value(np.array(res.cpg_dual.d19).reshape(6))
    prob.constraints[20].save_dual_value(np.array(res.cpg_dual.d20).reshape(6))
    prob.constraints[21].save_dual_value(np.array(res.cpg_dual.d21).reshape(6))
    prob.constraints[22].save_dual_value(np.array(res.cpg_dual.d22).reshape(6))
    prob.constraints[23].save_dual_value(np.array(res.cpg_dual.d23).reshape(6))
    prob.constraints[24].save_dual_value(np.array(res.cpg_dual.d24).reshape(6))
    prob.constraints[25].save_dual_value(np.array(res.cpg_dual.d25).reshape(6))
    prob.constraints[26].save_dual_value(np.array(res.cpg_dual.d26).reshape(6))
    prob.constraints[27].save_dual_value(np.array(res.cpg_dual.d27).reshape(6))
    prob.constraints[28].save_dual_value(np.array(res.cpg_dual.d28).reshape(6))
    prob.constraints[29].save_dual_value(np.array(res.cpg_dual.d29).reshape(6))
    prob.constraints[30].save_dual_value(np.array(res.cpg_dual.d30).reshape(6))
    prob.constraints[31].save_dual_value(np.array(res.cpg_dual.d31).reshape(6))
    prob.constraints[32].save_dual_value(np.array(res.cpg_dual.d32))
    prob.constraints[33].save_dual_value(np.array(res.cpg_dual.d33))
    prob.constraints[34].save_dual_value(np.array(res.cpg_dual.d34))
    prob.constraints[35].save_dual_value(np.array(res.cpg_dual.d35))
    prob.constraints[36].save_dual_value(np.array(res.cpg_dual.d36))
    prob.constraints[37].save_dual_value(np.array(res.cpg_dual.d37))
    prob.constraints[38].save_dual_value(np.array(res.cpg_dual.d38))
    prob.constraints[39].save_dual_value(np.array(res.cpg_dual.d39))
    prob.constraints[40].save_dual_value(np.array(res.cpg_dual.d40))
    prob.constraints[41].save_dual_value(np.array(res.cpg_dual.d41))
    prob.constraints[42].save_dual_value(np.array(res.cpg_dual.d42))
    prob.constraints[43].save_dual_value(np.array(res.cpg_dual.d43))
    prob.constraints[44].save_dual_value(np.array(res.cpg_dual.d44))
    prob.constraints[45].save_dual_value(np.array(res.cpg_dual.d45))
    prob.constraints[46].save_dual_value(np.array(res.cpg_dual.d46))
    prob.constraints[47].save_dual_value(np.array(res.cpg_dual.d47))
    prob.constraints[48].save_dual_value(np.array(res.cpg_dual.d48))
    prob.constraints[49].save_dual_value(np.array(res.cpg_dual.d49))
    prob.constraints[50].save_dual_value(np.array(res.cpg_dual.d50))
    prob.constraints[51].save_dual_value(np.array(res.cpg_dual.d51))
    prob.constraints[52].save_dual_value(np.array(res.cpg_dual.d52))
    prob.constraints[53].save_dual_value(np.array(res.cpg_dual.d53))
    prob.constraints[54].save_dual_value(np.array(res.cpg_dual.d54))
    prob.constraints[55].save_dual_value(np.array(res.cpg_dual.d55))
    prob.constraints[56].save_dual_value(np.array(res.cpg_dual.d56))
    prob.constraints[57].save_dual_value(np.array(res.cpg_dual.d57))
    prob.constraints[58].save_dual_value(np.array(res.cpg_dual.d58))
    prob.constraints[59].save_dual_value(np.array(res.cpg_dual.d59))
    prob.constraints[60].save_dual_value(np.array(res.cpg_dual.d60))
    prob.constraints[61].save_dual_value(np.array(res.cpg_dual.d61))
    prob.constraints[62].save_dual_value(np.array(res.cpg_dual.d62))
    prob.constraints[63].save_dual_value(np.array(res.cpg_dual.d63).reshape(3))
    prob.constraints[64].save_dual_value(np.array(res.cpg_dual.d64).reshape(3))
    prob.constraints[65].save_dual_value(np.array(res.cpg_dual.d65).reshape(3))
    prob.constraints[66].save_dual_value(np.array(res.cpg_dual.d66).reshape(3))
    prob.constraints[67].save_dual_value(np.array(res.cpg_dual.d67).reshape(3))
    prob.constraints[68].save_dual_value(np.array(res.cpg_dual.d68).reshape(3))
    prob.constraints[69].save_dual_value(np.array(res.cpg_dual.d69).reshape(3))
    prob.constraints[70].save_dual_value(np.array(res.cpg_dual.d70).reshape(3))
    prob.constraints[71].save_dual_value(np.array(res.cpg_dual.d71).reshape(3))
    prob.constraints[72].save_dual_value(np.array(res.cpg_dual.d72).reshape(3))
    prob.constraints[73].save_dual_value(np.array(res.cpg_dual.d73).reshape(3))
    prob.constraints[74].save_dual_value(np.array(res.cpg_dual.d74).reshape(3))
    prob.constraints[75].save_dual_value(np.array(res.cpg_dual.d75).reshape(3))
    prob.constraints[76].save_dual_value(np.array(res.cpg_dual.d76).reshape(3))
    prob.constraints[77].save_dual_value(np.array(res.cpg_dual.d77).reshape(3))
    prob.constraints[78].save_dual_value(np.array(res.cpg_dual.d78).reshape(3))
    prob.constraints[79].save_dual_value(np.array(res.cpg_dual.d79).reshape(3))
    prob.constraints[80].save_dual_value(np.array(res.cpg_dual.d80).reshape(3))
    prob.constraints[81].save_dual_value(np.array(res.cpg_dual.d81).reshape(3))
    prob.constraints[82].save_dual_value(np.array(res.cpg_dual.d82).reshape(3))
    prob.constraints[83].save_dual_value(np.array(res.cpg_dual.d83).reshape(3))
    prob.constraints[84].save_dual_value(np.array(res.cpg_dual.d84).reshape(3))
    prob.constraints[85].save_dual_value(np.array(res.cpg_dual.d85).reshape(3))
    prob.constraints[86].save_dual_value(np.array(res.cpg_dual.d86).reshape(3))
    prob.constraints[87].save_dual_value(np.array(res.cpg_dual.d87).reshape(3))
    prob.constraints[88].save_dual_value(np.array(res.cpg_dual.d88).reshape(3))
    prob.constraints[89].save_dual_value(np.array(res.cpg_dual.d89).reshape(3))
    prob.constraints[90].save_dual_value(np.array(res.cpg_dual.d90).reshape(3))
    prob.constraints[91].save_dual_value(np.array(res.cpg_dual.d91).reshape(3))
    prob.constraints[92].save_dual_value(np.array(res.cpg_dual.d92).reshape(3))
    prob.constraints[93].save_dual_value(np.array(res.cpg_dual.d93).reshape(3))
    prob.constraints[94].save_dual_value(np.array(res.cpg_dual.d94).reshape(3))
    prob.constraints[95].save_dual_value(np.array(res.cpg_dual.d95).reshape(3))
    prob.constraints[96].save_dual_value(np.array(res.cpg_dual.d96).reshape(3))
    prob.constraints[97].save_dual_value(np.array(res.cpg_dual.d97).reshape(3))
    prob.constraints[98].save_dual_value(np.array(res.cpg_dual.d98).reshape(3))
    prob.constraints[99].save_dual_value(np.array(res.cpg_dual.d99).reshape(3))
    prob.constraints[100].save_dual_value(np.array(res.cpg_dual.d100).reshape(3))
    prob.constraints[101].save_dual_value(np.array(res.cpg_dual.d101).reshape(3))
    prob.constraints[102].save_dual_value(np.array(res.cpg_dual.d102).reshape(3))
    prob.constraints[103].save_dual_value(np.array(res.cpg_dual.d103).reshape(3))
    prob.constraints[104].save_dual_value(np.array(res.cpg_dual.d104).reshape(3))
    prob.constraints[105].save_dual_value(np.array(res.cpg_dual.d105).reshape(3))
    prob.constraints[106].save_dual_value(np.array(res.cpg_dual.d106).reshape(3))
    prob.constraints[107].save_dual_value(np.array(res.cpg_dual.d107).reshape(3))
    prob.constraints[108].save_dual_value(np.array(res.cpg_dual.d108).reshape(3))
    prob.constraints[109].save_dual_value(np.array(res.cpg_dual.d109).reshape(3))
    prob.constraints[110].save_dual_value(np.array(res.cpg_dual.d110).reshape(3))
    prob.constraints[111].save_dual_value(np.array(res.cpg_dual.d111).reshape(3))
    prob.constraints[112].save_dual_value(np.array(res.cpg_dual.d112).reshape(3))
    prob.constraints[113].save_dual_value(np.array(res.cpg_dual.d113).reshape(3))
    prob.constraints[114].save_dual_value(np.array(res.cpg_dual.d114).reshape(3))
    prob.constraints[115].save_dual_value(np.array(res.cpg_dual.d115).reshape(3))
    prob.constraints[116].save_dual_value(np.array(res.cpg_dual.d116).reshape(3))
    prob.constraints[117].save_dual_value(np.array(res.cpg_dual.d117).reshape(3))
    prob.constraints[118].save_dual_value(np.array(res.cpg_dual.d118).reshape(3))
    prob.constraints[119].save_dual_value(np.array(res.cpg_dual.d119).reshape(3))
    prob.constraints[120].save_dual_value(np.array(res.cpg_dual.d120).reshape(3))
    prob.constraints[121].save_dual_value(np.array(res.cpg_dual.d121).reshape(3))
    prob.constraints[122].save_dual_value(np.array(res.cpg_dual.d122).reshape(3))
    prob.constraints[123].save_dual_value(np.array(res.cpg_dual.d123).reshape(3))
    prob.constraints[124].save_dual_value(np.array(res.cpg_dual.d124).reshape(3))

    # store additional solver information in problem object
    prob._status = res.cpg_info.status
    if abs(res.cpg_info.obj_val) == 1e30:
        prob._value = np.sign(res.cpg_info.obj_val) * np.inf
    else:
        prob._value = res.cpg_info.obj_val
    primal_vars = {var.id: var.value for var in prob.variables()}
    dual_vars = {c.id: c.dual_value for c in prob.constraints}
    solver_specific_stats = {'obj_val': res.cpg_info.obj_val,
                             'status': prob._status,
                             'iter': res.cpg_info.iter,
                             'pri_res': res.cpg_info.pri_res,
                             'dua_res': res.cpg_info.dua_res,
                             'time': res.cpg_info.time}
    attr = {'solve_time': t1 - t0, 'solver_specific_stats': solver_specific_stats, 'num_iters': res.cpg_info.iter}
    prob._solution = Solution(prob.status, prob.value, primal_vars, dual_vars, attr)
    results_dict = {'solver_specific_stats': solver_specific_stats,
                    'num_iters': res.cpg_info.iter,
                    'solve_time': t1 - t0}
    prob._solver_stats = SolverStats(results_dict, 'SCS')

    return prob.value
