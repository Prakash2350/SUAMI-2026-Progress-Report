import math

from firedrake import *

from . import config
from .output import make_vtk_file, write_vtk_frame
from .problem import build_problem


def solve_one_mesh(
    N,
    anchoring,
    mode="direct",
    save_vtk=True,
    exact_case_name=None,
    problem_mode=None,
):
    problem = build_problem(N, anchoring, exact_case_name, problem_mode)

    if mode == "direct":
        return _solve_direct(problem, save_vtk)

    if mode == "gradient":
        return _solve_gradient(problem, save_vtk)

    raise ValueError("mode must be 'direct' or 'gradient'")


def _result(problem, L2_err, H1_err, vtk_path, energy_history=None, update_history=None):
    return {
        "N": problem.N,
        "h": problem.h,
        "exact_case": problem.exact_case_name,
        "anchoring": problem.anchoring,
        "L2": L2_err,
        "H1": H1_err,
        "energy_history": energy_history or [],
        "update_history": update_history or [],
        "vtk_path": vtk_path,
    }


def _solve_direct(problem, save_vtk):
    Q_tensor = problem.Q_tensor
    q = problem.q
    Q = Q_tensor(q)
    P = Q_tensor(problem.p)

    bulk_Q = (
        -problem.a2 * Q
        - problem.a3 * dot(Q, Q)
        + problem.a4 * inner(Q, Q) * Q
    )

    F_direct = (
        problem.l1 * inner(grad(Q), grad(P)) * dx
        + (1.0 / problem.eta**2) * inner(bulk_Q, P) * dx
        + inner(problem.surface_grad(Q), P) * ds
    )

    if problem.problem_mode == "mms":
        F_direct += (
            -inner(problem.f_tensor, P) * dx
            - inner(problem.boundary_rhs, P) * ds
        )

    if problem.problem_mode == "mms":
        # Start near the manufactured solution so Newton selects the MMS branch.
        q.interpolate(problem.q_exact)
    else:
        q.assign(0.0)

    solve(
        F_direct == 0,
        q,
        solver_parameters=config.DIRECT_SOLVER_PARAMETERS.copy(),
    )

    vtk_file, vtk_path = make_vtk_file(
        "direct",
        problem.anchoring,
        problem.N,
        exact_case_name=problem.exact_case_name,
        save_vtk=save_vtk,
    )
    write_vtk_frame(
        vtk_file,
        q,
        problem.q_exact_fn,
        problem.q_error,
    )

    L2_err, H1_err = problem.compute_errors(q)
    return _result(problem, L2_err, H1_err, vtk_path)


def _solve_gradient(problem, save_vtk):
    Q_tensor = problem.Q_tensor
    time_step = Constant(config.gradient_time_step_value)

    q_old = Function(problem.V, name="q")
    q_new = Function(problem.V, name="q_new")

    q_old.assign(0.0)
    q_new.assign(q_old)

    Q_old = Q_tensor(q_old)
    Q_new = Q_tensor(q_new)
    P = Q_tensor(problem.p)

    bulk_new = (
        -problem.a2 * Q_new
        - problem.a3 * dot(Q_new, Q_new)
        + problem.a4 * inner(Q_new, Q_new) * Q_new
    )

    F_gd = (
        inner((Q_new - Q_old) / time_step, P) * dx
        + problem.l1 * inner(grad(Q_new), grad(P)) * dx
        + (1.0 / problem.eta**2) * inner(bulk_new, P) * dx
        + inner(problem.surface_grad(Q_new), P) * ds
    )

    if problem.problem_mode == "mms":
        F_gd += (
            -inner(problem.f_tensor, P) * dx
            - inner(problem.boundary_rhs, P) * ds
        )

    vtk_file, vtk_path = make_vtk_file(
        "gradient",
        problem.anchoring,
        problem.N,
        exact_case_name=problem.exact_case_name,
        save_vtk=save_vtk,
    )
    write_vtk_frame(
        vtk_file,
        q_old,
        problem.q_exact_fn,
        problem.q_error,
        time_value=0.0,
    )

    energy_history = []
    update_history = []

    E0 = float(assemble(problem.energy_form(Q_old)))
    energy_history.append(E0)

    for k in range(1, config.gradient_max_iter + 1):
        q_new.assign(q_old)

        solve(
            F_gd == 0,
            q_new,
            solver_parameters=config.GRADIENT_SOLVER_PARAMETERS.copy(),
        )

        Q_update = Q_tensor(q_new) - Q_tensor(q_old)
        update_norm = math.sqrt(float(assemble(inner(Q_update, Q_update) * dx)))

        q_old.assign(q_new)

        energy_value = float(assemble(problem.energy_form(Q_tensor(q_old))))
        energy_history.append(energy_value)
        update_history.append(update_norm)

        if k % config.gradient_save_every == 0:
            write_vtk_frame(
                vtk_file,
                q_old,
                problem.q_exact_fn,
                problem.q_error,
                time_value=k * config.gradient_time_step_value,
            )

        if k % 50 == 0 and problem.problem_mode == "mms":
            L2_err, H1_err = problem.compute_errors(q_old)
            print(
                f"[{problem.anchoring}] iter={k:5d}, "
                f"energy={energy_value:.12e}, "
                f"update={update_norm:.3e}, "
                f"L2={L2_err:.3e}, "
                f"H1={H1_err:.3e}"
            )
        elif k % 50 == 0:
            print(
                f"[{problem.anchoring}] iter={k:5d}, "
                f"energy={energy_value:.12e}, "
                f"update={update_norm:.3e}"
            )

        if update_norm < config.gradient_tol:
            print(
                f"[{problem.anchoring}] converged at iter={k}, "
                f"update={update_norm:.3e}"
            )
            write_vtk_frame(
                vtk_file,
                q_old,
                problem.q_exact_fn,
                problem.q_error,
                time_value=k * config.gradient_time_step_value,
            )
            break

    problem.q.assign(q_old)

    L2_err, H1_err = problem.compute_errors(problem.q)
    return _result(
        problem,
        L2_err,
        H1_err,
        vtk_path,
        energy_history=energy_history,
        update_history=update_history,
    )
