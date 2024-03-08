import os
import scipy.sparse as spa
import numpy as np
import torch

import proxsuite

from torch.autograd import Function
from .utils import expandParam, extract_nBatch, bger


def QPFunction(
    eps=1e-9,
    maxIter=1000,
    eps_backward=1.0e-4,
    rho_backward=1.0e-6,
    mu_backward=1.0e-6,
    omp_parallel=False,
    structural_feasibility=True,
):
    """
    Solve a batch of Quadratic Programming (QP) problems.

    This function solves QP problems of the form:
        min 0.5*z'*Q*z + p'*z
        s.t. l <= G*z <= h
             A*z = b

    The QP can be infeasible - in this case the solver will return a solution to
    the closest feasible QP.

    Args:
        eps (float, optional): Tolerance for the primal infeasibility. Defaults to 1e-9.
        maxIter (int, optional): Maximum number of iterations. Defaults to 1000.
        eps_backward (float, optional): Tolerance for the backward pass. Defaults to 1e-4.
        rho_backward (float, optional): The new value for the primal proximal parameter. Defaults to 1e-6.
        mu_backward (float, optional): The new dual proximal parameter used for both equality and inequality. Defaults to 1e-6.
        omp_parallel (bool, optional): Whether to solve the QP in parallel. Requires that proxsuite is compiled with openmp support. Defaults to False.
        structural_feasibility (bool, optional): Whether to solve the QP with structural feasibility. Defaults to True.

    Returns:
        QPFunctionFn or QPFunctionFn_infeas: A callable object that represents the QP problem solver.
        We disinguish two cases:
            1. The QP is feasible. In this case, we solve the QP problem.
            2. The QP is infeasible. In this case, we solve the closest feasible QP problem.

    The callable object has two main methods:

    Forward:
        Solve the QP problem.

        Args:
            Q (torch.Tensor): Batch of quadratic cost matrices of size (nBatch, n, n) or (n, n).
            p (torch.Tensor): Batch of linear cost vectors of size (nBatch, n) or (n).
            A (torch.Tensor, optional): Batch of eq. constraint matrices of size (nBatch, p, n) or (p, n).
            b (torch.Tensor, optional): Batch of eq. constraint vectors of size (nBatch, p) or (p).
            G (torch.Tensor): Batch of ineq. constraint matrices of size (nBatch, m, n) or (m, n).
            l (torch.Tensor): Batch of ineq. lower bound vectors of size (nBatch, m) or (m).
            u (torch.Tensor): Batch of ineq. upper bound vectors of size (nBatch, m) or (m).

        Returns:
            zhats (torch.Tensor): Batch of optimal primal solutions of size (nBatch, n).
            lams (torch.Tensor): Batch of dual variables for eq. constraint of size (nBatch, m).
            nus (torch.Tensor): Batch of dual variables  for ineq. constraints of size (nBatch, p).
            Only for infeasible case:
                s_e (torch.Tensor): Batch of slack variables for eq. constraints of size (nBatch, m).
                s_i (torch.Tensor): Batch of slack variables for ineq. constraints of size (nBatch, p).

    Backward:
        Compute the gradients of the QP problem wrt its parameters.

        Args:
            dl_dzhat (torch.Tensor): Batch of gradients of size (nBatch, n).
            dl_dlams (torch.Tensor, optional): Batch of gradients of size (nBatch, p).
            dl_dnus (torch.Tensor, optional): Batch of gradients of size (nBatch, m).
            Only for infeasible case:
                dl_ds_e (torch.Tensor, optional): Batch of gradients of size (nBatch, m).
                dl_ds_i (torch.Tensor, optional): Batch of gradients of size (nBatch, m).

        Returns:
            dQs (torch.Tensor): Batch of gradients of size (nBatch, n, n).
            dps (torch.Tensor): Batch of gradients of size (nBatch, n).
            dAs (torch.Tensor): Batch of gradients of size (nBatch, p, n).
            dbs (torch.Tensor): Batch of gradients of size (nBatch, p).
            dGs (torch.Tensor): Batch of gradients of size (nBatch, m, n).
            dls (torch.Tensor): Batch of gradients of size (nBatch, m).
            dus (torch.Tensor): Batch of gradients of size (nBatch, m).
    """
    global proxqp_parallel
    proxqp_parallel = omp_parallel

    class QPFunctionFn(Function):
        @staticmethod
        def forward(ctx, Q_, p_, A_, b_, G_, l_, u_):
            nBatch = extract_nBatch(Q_, p_, A_, b_, G_, l_, u_)
            Q, _ = expandParam(Q_, nBatch, 3)
            p, _ = expandParam(p_, nBatch, 2)
            G, _ = expandParam(G_, nBatch, 3)
            u, _ = expandParam(u_, nBatch, 2)
            l, _ = expandParam(l_, nBatch, 2)
            A, _ = expandParam(A_, nBatch, 3)
            b, _ = expandParam(b_, nBatch, 2)

            ctx.vector_of_qps = proxsuite.proxqp.dense.BatchQP()

            ctx.nBatch = nBatch

            _, nineq, nz = G.size()
            neq = A.size(1) if A.nelement() > 0 else 0
            assert neq > 0 or nineq > 0
            ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz

            ctx.cpu = os.cpu_count()
            if ctx.cpu is not None:
                ctx.cpu = max(1, int(ctx.cpu / 2))

            zhats = torch.empty((nBatch, ctx.nz), dtype=Q.dtype)
            lams = torch.empty((nBatch, ctx.neq), dtype=Q.dtype)
            nus = torch.empty((nBatch, ctx.nineq), dtype=Q.dtype)

            for i in range(nBatch):
                qp = ctx.vector_of_qps.init_qp_in_place(ctx.nz, ctx.neq, ctx.nineq)
                qp.settings.primal_infeasibility_solving = False
                qp.settings.max_iter = maxIter
                qp.settings.max_iter_in = 100
                default_rho = 5.0e-5
                qp.settings.default_rho = default_rho
                qp.settings.refactor_rho_threshold = default_rho  # no refactorization
                qp.settings.eps_abs = eps
                Ai, bi = (A[i], b[i]) if neq > 0 else (None, None)
                H__ = None
                if Q[i] is not None:
                    H__ = Q[i].cpu().numpy()
                p__ = None
                if p[i] is not None:
                    p__ = p[i].cpu().numpy()
                G__ = None
                if G[i] is not None:
                    G__ = G[i].cpu().numpy()
                u__ = None
                if u[i] is not None:
                    u__ = u[i].cpu().numpy()
                l__ = None
                if l[i] is not None:
                    l__ = l[i].cpu().numpy()
                A__ = None
                if Ai is not None:
                    A__ = Ai.cpu().numpy()
                b__ = None
                if bi is not None:
                    b__ = bi.cpu().numpy()

                qp.init(
                    H=H__, g=p__, A=A__, b=b__, C=G__, l=l__, u=u__, rho=default_rho
                )

            if proxqp_parallel:
                proxsuite.proxqp.dense.solve_in_parallel(
                    num_threads=ctx.cpu, qps=ctx.vector_of_qps
                )
            else:
                for i in range(ctx.vector_of_qps.size()):
                    ctx.vector_of_qps.get(i).solve()

            for i in range(nBatch):
                zhats[i] = torch.tensor(ctx.vector_of_qps.get(i).results.x)
                lams[i] = torch.tensor(ctx.vector_of_qps.get(i).results.y)
                nus[i] = torch.tensor(ctx.vector_of_qps.get(i).results.z)

            return zhats, lams, nus

        @staticmethod
        def backward(ctx, dl_dzhat, dl_dlams, dl_dnus):
            device = dl_dzhat.device
            nBatch, dim, neq, nineq = ctx.nBatch, ctx.nz, ctx.neq, ctx.nineq
            dQs = torch.empty(nBatch, ctx.nz, ctx.nz, device=device)
            dps = torch.empty(nBatch, ctx.nz, device=device)
            dGs = torch.empty(nBatch, ctx.nineq, ctx.nz, device=device)
            dus = torch.empty(nBatch, ctx.nineq, device=device)
            dls = torch.empty(nBatch, ctx.nineq, device=device)
            dAs = torch.empty(nBatch, ctx.neq, ctx.nz, device=device)
            dbs = torch.empty(nBatch, ctx.neq, device=device)

            ctx.cpu = os.cpu_count()
            if ctx.cpu is not None:
                ctx.cpu = max(1, int(ctx.cpu / 2))

            n_tot = dim + neq + nineq

            if proxqp_parallel:
                vector_of_loss_derivatives = (
                    proxsuite.proxqp.dense.VectorLossDerivatives()
                )

                for i in range(nBatch):
                    rhs = np.zeros(n_tot)
                    rhs[:dim] = dl_dzhat[i]
                    if dl_dlams != None:
                        rhs[dim : dim + neq] = dl_dlams[i]
                    if dl_dnus != None:
                        rhs[dim + neq :] = dl_dnus[i]
                    vector_of_loss_derivatives.append(rhs)

                proxsuite.proxqp.dense.solve_backward_in_parallel(
                    num_threads=ctx.cpu,
                    qps=ctx.vector_of_qps,
                    loss_derivatives=vector_of_loss_derivatives,
                    eps=eps_backward,
                    rho_backward=rho_backward,
                    mu_backward=mu_backward,
                )  # try with systematic fwd bwd
            else:
                for i in range(nBatch):
                    rhs = np.zeros(n_tot)
                    rhs[:dim] = dl_dzhat[i].cpu()
                    if dl_dlams != None:
                        rhs[dim : dim + neq] = dl_dlams[i].cpu()
                    if dl_dnus != None:
                        rhs[dim + neq :] = dl_dnus[i].cpu()
                    qpi = ctx.vector_of_qps.get(i)
                    proxsuite.proxqp.dense.compute_backward(
                        qp=qpi,
                        loss_derivative=rhs,
                        eps=eps_backward,
                        rho_backward=rho_backward,
                        mu_backward=mu_backward,
                    )

            for i in range(nBatch):
                dQs[i] = torch.tensor(
                    ctx.vector_of_qps.get(i).model.backward_data.dL_dH
                )
                dps[i] = torch.tensor(
                    ctx.vector_of_qps.get(i).model.backward_data.dL_dg
                )
                dGs[i] = torch.tensor(
                    ctx.vector_of_qps.get(i).model.backward_data.dL_dC
                )
                dus[i] = torch.tensor(
                    ctx.vector_of_qps.get(i).model.backward_data.dL_du
                )
                dls[i] = torch.tensor(
                    ctx.vector_of_qps.get(i).model.backward_data.dL_dl
                )
                dAs[i] = torch.tensor(
                    ctx.vector_of_qps.get(i).model.backward_data.dL_dA
                )
                dbs[i] = torch.tensor(
                    ctx.vector_of_qps.get(i).model.backward_data.dL_db
                )

            grads = (dQs, dps, dAs, dbs, dGs, dls, dus)

            return grads

    class QPFunctionFn_infeas(Function):
        @staticmethod
        def forward(ctx, Q_, p_, A_, b_, G_, l_, u_):
            nBatch = extract_nBatch(Q_, p_, A_, b_, G_, l_, u_)

            Q, _ = expandParam(Q_, nBatch, 3)
            p, _ = expandParam(p_, nBatch, 2)
            G, _ = expandParam(G_, nBatch, 3)
            u, _ = expandParam(u_, nBatch, 2)
            l, _ = expandParam(l_, nBatch, 2)
            A, _ = expandParam(A_, nBatch, 3)
            b, _ = expandParam(b_, nBatch, 2)

            h = torch.cat((-l, u), axis=1)  # single-sided inequality
            G = torch.cat((-G, G), axis=1)  # single-sided inequality

            _, nineq, nz = G.size()
            neq = A.size(1) if A.nelement() > 0 else 0
            assert neq > 0 or nineq > 0
            ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz

            zhats = torch.empty((nBatch, ctx.nz), dtype=Q.dtype)
            nus = torch.empty((nBatch, ctx.nineq), dtype=Q.dtype)
            lams = (
                torch.empty(nBatch, ctx.neq, dtype=Q.dtype)
                if ctx.neq > 0
                else torch.empty()
            )
            s_e = (
                torch.empty(nBatch, ctx.neq, dtype=Q.dtype)
                if ctx.neq > 0
                else torch.empty()
            )
            slacks = torch.empty((nBatch, ctx.nineq), dtype=Q.dtype)
            s_i = torch.empty((nBatch, ctx.nineq), dtype=Q.dtype)

            vector_of_qps = proxsuite.proxqp.dense.BatchQP()

            ctx.cpu = os.cpu_count()
            if ctx.cpu is not None:
                ctx.cpu = max(1, int(ctx.cpu / 2))
            l = -np.ones(ctx.nineq) * 1.0e20

            for i in range(nBatch):
                qp = vector_of_qps.init_qp_in_place(ctx.nz, ctx.neq, ctx.nineq)
                qp.settings.primal_infeasibility_solving = True
                qp.settings.max_iter = maxIter
                qp.settings.max_iter_in = 100
                default_rho = 5.0e-5
                qp.settings.default_rho = default_rho
                qp.settings.refactor_rho_threshold = default_rho  # no refactorization
                qp.settings.eps_abs = eps
                Ai, bi = (A[i], b[i]) if neq > 0 else (None, None)
                H__ = None
                if Q[i] is not None:
                    H__ = Q[i].cpu().numpy()
                p__ = None
                if p[i] is not None:
                    p__ = p[i].cpu().numpy()
                G__ = None
                if G[i] is not None:
                    G__ = G[i].cpu().numpy()
                u__ = None
                if h[i] is not None:
                    u__ = h[i].cpu().numpy()
                # l__ = None
                # if (l[i] is not None):
                #     l__ = l[i].cpu().numpy()
                A__ = None
                if Ai is not None:
                    A__ = Ai.cpu().numpy()
                b__ = None
                if bi is not None:
                    b__ = bi.cpu().numpy()

                qp.init(H=H__, g=p__, A=A__, b=b__, C=G__, l=l, u=u__, rho=default_rho)

            if proxqp_parallel:
                proxsuite.proxqp.dense.solve_in_parallel(
                    num_threads=ctx.cpu, qps=vector_of_qps
                )
            else:
                for i in range(vector_of_qps.size()):
                    vector_of_qps.get(i).solve()

            for i in range(nBatch):
                si = -h[i] + G[i] @ vector_of_qps.get(i).results.x
                zhats[i] = torch.tensor(vector_of_qps.get(i).results.x)
                nus[i] = torch.tensor(vector_of_qps.get(i).results.z)
                slacks[i] = si.clone().detach()
                if neq > 0:
                    lams[i] = torch.tensor(vector_of_qps.get(i).results.y)
                    s_e[i] = torch.tensor(vector_of_qps.get(i).results.se)
                s_i[i] = torch.tensor(vector_of_qps.get(i).results.si)

            ctx.lams = lams
            ctx.nus = nus
            ctx.slacks = slacks
            ctx.save_for_backward(zhats, s_e, Q_, p_, G_, l_, u_, A_, b_)
            return zhats, lams, nus, s_e, s_i

        @staticmethod
        def backward(ctx, dl_dzhat, dl_dlams, dl_dnus, dl_ds_e, dl_ds_i):
            zhats, s_e, Q, p, G, l, u, A, b = ctx.saved_tensors
            nBatch = extract_nBatch(Q, p, A, b, G, l, u)

            Q, Q_e = expandParam(Q, nBatch, 3)
            p, p_e = expandParam(p, nBatch, 2)
            G, G_e = expandParam(G, nBatch, 3)
            _, u_e = expandParam(u, nBatch, 2)
            _, l_e = expandParam(l, nBatch, 2)
            A, A_e = expandParam(A, nBatch, 3)
            b, b_e = expandParam(b, nBatch, 2)

            h_e = l_e or u_e
            G = torch.cat((-G, G), axis=1)

            neq, nineq = ctx.neq, ctx.nineq
            dx = torch.zeros((nBatch, Q.shape[1]))
            dnu = None
            b_5 = None
            dlam = None
            if nineq > 0:
                dnu = torch.zeros((nBatch, nineq))
            if neq > 0:
                dlam = torch.zeros((nBatch, neq))
                b_5 = torch.zeros((nBatch, Q.shape[1]))

            b_6 = torch.zeros((nBatch, Q.shape[1]))
            P_2_c_s_i = torch.zeros((nBatch, nineq))

            n_row = Q.shape[1] + 2 * nineq
            n_col = 2 * Q.shape[1] + 2 * nineq
            if neq > 0:
                n_col += neq + Q.shape[1]
                n_row += 2 * neq
            ctx.cpu = os.cpu_count()
            if ctx.cpu is not None:
                ctx.cpu = max(1, int(ctx.cpu / 2))

            kkt = np.zeros((n_row, n_col))
            vector_of_qps = proxsuite.proxqp.sparse.BatchQP()

            for i in range(nBatch):
                Q_i = Q[i].numpy()
                C_i = G[i].numpy()
                A_i = None
                if A is not None:
                    if A.shape[0] != 0:
                        A_i = A[i].numpy()
                z_i = ctx.nus[i]
                s_i = ctx.slacks[i]  # G @ z_- h = slacks

                dim = Q_i.shape[0]
                n_eq = neq
                n_in = nineq

                P_1 = np.minimum(s_i, 0.0) + z_i >= 0.0
                P_2 = s_i <= 0.0
                P_2_c_s_i[i] = np.maximum(
                    s_i, 0.0
                )  # keep only (1-P_2)s_i for backward calculation afterward

                kkt[:dim, :dim] = Q_i
                if neq > 0:
                    kkt[:dim, dim : dim + n_eq] = A_i.transpose()
                    kkt[dim : dim + n_eq, :dim] = A_i
                    kkt[dim + n_eq + n_in : dim + 2 * n_eq + n_in, dim : dim + n_eq] = (
                        -np.eye(n_eq)
                    )
                    kkt[
                        dim + n_eq + n_in : dim + 2 * n_eq + n_in,
                        dim + n_eq + 2 * n_in : 2 * dim + n_eq + 2 * n_in,
                    ] = A_i

                kkt[:dim, dim + n_eq : dim + n_eq + n_in] = C_i.transpose()
                kkt[dim + n_eq : dim + n_eq + n_in, :dim] = C_i

                D_1_c = np.eye(n_in)  # represents [s_i]_- + z_i < 0
                D_1_c[P_1, P_1] = 0.0
                D_1 = np.eye(n_in) - D_1_c  # represents [s_i]_- + z_i >= 0
                D_2_c = np.eye(n_in)  # represents s_i > 0
                D_2_c[P_2, P_2] = 0.0
                D_2 = np.eye(n_in) - D_2_c  # represents s_i <= 0
                kkt[dim + 2 * n_eq + n_in :, dim + n_eq : dim + n_eq + n_in] = -np.eye(
                    n_in
                )
                kkt[
                    dim + n_eq : dim + n_eq + n_in,
                    dim + n_eq + n_in : dim + n_eq + 2 * n_in,
                ] = D_1_c
                kkt[
                    dim + 2 * n_eq + n_in :, dim + n_eq + n_in : dim + n_eq + 2 * n_in
                ] = -np.multiply(np.diag(D_1)[:, None], D_2)
                dim_ = 0
                if n_eq > 0:
                    dim_ += dim
                kkt[dim + 2 * n_eq + n_in :, dim + n_eq + 2 * n_in + dim_ :] = (
                    np.multiply(np.diag(D_2_c)[:, None], C_i)
                )

                rhs = np.zeros(kkt.shape[0])
                rhs[:dim] = -dl_dzhat[i]
                if dl_dlams != None:
                    rhs[dim : dim + n_eq] = -dl_dlams[i]
                if dl_dnus != None:
                    rhs[dim + n_eq : dim + n_eq + n_in] = -dl_dnus[i]
                if dl_ds_e != None:
                    if dl_ds_e.shape[0] != 0:
                        rhs[dim + n_eq + n_in : dim + 2 * n_eq + n_in] = -dl_ds_e[i]
                if dl_ds_i != None:
                    if dl_ds_i.shape[0] != 0:
                        rhs[dim + 2 * n_eq + n_in :] = -dl_ds_i[i]

                l = np.zeros(0)
                u = np.zeros(0)

                C = spa.csc_matrix((0, n_col))
                H = spa.csc_matrix((n_col, n_col))
                g = np.zeros(n_col)

                qp = vector_of_qps.init_qp_in_place(
                    H.shape[0], kkt.shape[0], C.shape[0]
                )

                qp.settings.primal_infeasibility_solving = True
                qp.settings.eps_abs = eps_backward
                qp.settings.max_iter = 10
                qp.settings.default_rho = 1.0e-3
                qp.settings.refactor_rho_threshold = 1.0e-3
                qp.init(
                    H,
                    g,
                    spa.csc_matrix(kkt),
                    rhs,
                    C,
                    l,
                    u,
                )

            if proxqp_parallel:
                proxsuite.proxqp.sparse.solve_in_parallel(
                    num_threads=ctx.cpu, qps=vector_of_qps
                )
            else:
                for i in range(vector_of_qps.size()):
                    vector_of_qps.get(i).solve()

            for i in range(nBatch):
                dx[i] = torch.from_numpy(
                    np.float64(vector_of_qps.get(i).results.x[:dim])
                )
                if n_eq > 0:
                    dlam[i] = torch.from_numpy(
                        np.float64(vector_of_qps.get(i).results.x[dim : dim + n_eq])
                    )
                dnu[i] = torch.from_numpy(
                    np.float64(
                        vector_of_qps.get(i).results.x[dim + n_eq : dim + n_eq + n_in]
                    )
                )
                dim_ = 0
                if n_eq > 0:
                    b_5[i] = torch.from_numpy(
                        np.float64(
                            vector_of_qps.get(i).results.x[
                                dim + n_eq + 2 * n_in : 2 * dim + n_eq + 2 * n_in
                            ]
                        )
                    )
                    dim_ += dim
                b_6[i] = torch.from_numpy(
                    np.float64(
                        vector_of_qps.get(i).results.x[dim + n_eq + 2 * n_in + dim_ :]
                    )
                )

            dps = dx
            dGs = (
                bger(dnu.double(), zhats.double())
                + bger(ctx.nus.double(), dx.double())
                + bger(P_2_c_s_i.double(), b_6.double())
            )
            if G_e:
                dGs = dGs.mean(0)
            dhs = -dnu
            if h_e:
                dhs = dhs.mean(0)
            if neq > 0:
                dAs = (
                    bger(dlam.double(), zhats.double())
                    + bger(ctx.lams.double(), dx.double())
                    + bger(s_e.double(), b_5.double())
                )
                dbs = -dlam
                if A_e:
                    dAs = dAs.mean(0)
                if b_e:
                    dbs = dbs.mean(0)
            else:
                dAs, dbs = None, None
            dQs = 0.5 * (
                bger(dx.double(), zhats.double()) + bger(zhats.double(), dx.double())
            )
            if Q_e:
                dQs = dQs.mean(0)
            if p_e:
                dps = dps.mean(0)

            grads = (dQs, dps, dAs, dbs, dGs[nineq:, :], -dhs[:nineq], dhs[nineq:])

            return grads

    if structural_feasibility:
        return QPFunctionFn.apply
    else:
        return QPFunctionFn_infeas.apply
