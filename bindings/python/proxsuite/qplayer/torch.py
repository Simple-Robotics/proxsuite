import os
import scipy.sparse as spa
import numpy as np
import torch

import proxsuite

from torch.autograd import Function
from .utils import expandParam, extract_nBatch, extract_nBatch_double_sided, bger


def QPFunction(eps=1e-9, maxIter=1000, eps_backward=1.0e-4, structual_feasibility=True):
    class QPFunctionFn(Function):
        @staticmethod
        def forward(ctx, Q_, p_, A_, b_, G_, l_, u_):
            nBatch = extract_nBatch_double_sided(Q_, p_, A_, b_, G_, l_, u_)
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

            zhats = torch.Tensor(nBatch, ctx.nz).type_as(Q)
            for i in range(nBatch):
                qp = ctx.vector_of_qps.init_qp_in_place(ctx.nz, ctx.neq, ctx.nineq)
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

            proxsuite.proxqp.dense.solve_in_parallel(
                num_threads=ctx.cpu, qps=ctx.vector_of_qps
            )
            for i in range(nBatch):
                zhats[i] = torch.Tensor(ctx.vector_of_qps.get(i).results.x)

            return zhats

        @staticmethod
        def backward(ctx, dl_dzhat):
            nBatch, dim, neq, nineq = ctx.nBatch, ctx.nz, ctx.neq, ctx.nineq
            dQs = torch.Tensor(nBatch, ctx.nz, ctx.nz)
            dps = torch.Tensor(nBatch, ctx.nz)
            dGs = torch.Tensor(nBatch, ctx.nineq, ctx.nz)
            dus = torch.Tensor(nBatch, ctx.nineq)
            dls = torch.Tensor(nBatch, ctx.nineq)
            dAs = torch.Tensor(nBatch, ctx.neq, ctx.nz)
            dbs = torch.Tensor(nBatch, ctx.neq)

            ctx.cpu = os.cpu_count()
            if ctx.cpu is not None:
                ctx.cpu = max(1, int(ctx.cpu / 2))

            n_tot = dim + neq + nineq  # max size
            vector_of_loss_derivatives = proxsuite.proxqp.dense.VectorLossDerivatives()

            for i in range(nBatch):
                rhs = np.zeros(n_tot)
                rhs[:dim] = -dl_dzhat[i]
                vector_of_loss_derivatives.append(rhs)

            proxsuite.proxqp.dense.solve_backward_in_parallel(
                num_threads=ctx.cpu,
                qps=ctx.vector_of_qps,
                loss_derivatives=vector_of_loss_derivatives,
                eps=eps_backward,
            )

            for i in range(nBatch):
                dQs[i] = torch.Tensor(
                    ctx.vector_of_qps.get(i).model.backward_data.dL_dH
                )
                dps[i] = torch.Tensor(
                    ctx.vector_of_qps.get(i).model.backward_data.dL_dg
                )
                dGs[i] = torch.Tensor(
                    ctx.vector_of_qps.get(i).model.backward_data.dL_dC
                )
                dus[i] = torch.Tensor(
                    ctx.vector_of_qps.get(i).model.backward_data.dL_du
                )
                dls[i] = torch.Tensor(
                    ctx.vector_of_qps.get(i).model.backward_data.dL_dl
                )
                dAs[i] = torch.Tensor(
                    ctx.vector_of_qps.get(i).model.backward_data.dL_dA
                )
                dbs[i] = torch.Tensor(
                    ctx.vector_of_qps.get(i).model.backward_data.dL_db
                )

            grads = (dQs, dps, dAs, dbs, dGs, dls, dus)

            return grads

    class QPFunctionFn_infeas(Function):
        @staticmethod
        def forward(ctx, Q_, p_, G_, h_, A_, b_):
            nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_)
            Q, _ = expandParam(Q_, nBatch, 3)
            p, _ = expandParam(p_, nBatch, 2)
            G, _ = expandParam(G_, nBatch, 3)
            h, _ = expandParam(h_, nBatch, 2)
            A, _ = expandParam(A_, nBatch, 3)
            b, _ = expandParam(b_, nBatch, 2)

            _, nineq, nz = G.size()
            neq = A.size(1) if A.nelement() > 0 else 0
            assert neq > 0 or nineq > 0
            ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz

            zhats = torch.Tensor(nBatch, ctx.nz).type_as(Q)
            lams = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
            nus = (
                torch.Tensor(nBatch, ctx.neq).type_as(Q)
                if ctx.neq > 0
                else torch.Tensor()
            )
            s_e = (
                torch.Tensor(nBatch, ctx.neq).type_as(Q)
                if ctx.neq > 0
                else torch.Tensor()
            )
            slacks = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
            s_i = torch.Tensor(nBatch, ctx.nineq).type_as(Q)

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

            proxsuite.proxqp.dense.solve_in_parallel(
                num_threads=ctx.cpu, qps=vector_of_qps
            )

            for i in range(nBatch):
                si = -h[i] + G[i] @ vector_of_qps.get(i).results.x
                zhats[i] = torch.Tensor(vector_of_qps.get(i).results.x)
                lams[i] = torch.Tensor(vector_of_qps.get(i).results.z)
                slacks[i] = torch.Tensor(si)
                if neq > 0:
                    nus[i] = torch.Tensor(vector_of_qps.get(i).results.y)
                    s_e[i] = torch.Tensor(vector_of_qps.get(i).results.se)
                s_i[i] = torch.Tensor(vector_of_qps.get(i).results.si)

            ctx.lams = lams
            ctx.nus = nus
            ctx.slacks = slacks
            ctx.save_for_backward(zhats, s_e, Q_, p_, G_, h_, A_, b_)
            return zhats, s_e, s_i

        @staticmethod
        def backward(ctx, dl_dzhat, dl_ds_e, dl_ds_i):
            zhats, s_e, Q, p, G, h, A, b = ctx.saved_tensors
            nBatch = extract_nBatch(Q, p, G, h, A, b)

            Q, Q_e = expandParam(Q, nBatch, 3)
            p, p_e = expandParam(p, nBatch, 2)
            G, G_e = expandParam(G, nBatch, 3)
            h, h_e = expandParam(h, nBatch, 2)
            A, A_e = expandParam(A, nBatch, 3)
            b, b_e = expandParam(b, nBatch, 2)

            neq, nineq = ctx.neq, ctx.nineq
            dx = torch.zeros((nBatch, Q.shape[1]))
            dnu = None
            b_5 = None
            dlam = torch.zeros((nBatch, nineq))
            if neq > 0:
                dnu = torch.zeros((nBatch, neq))
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
                z_i = ctx.lams[i]
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
                    kkt[
                        dim + n_eq + n_in : dim + 2 * n_eq + n_in, dim : dim + n_eq
                    ] = -np.eye(n_eq)
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
                kkt[
                    dim + 2 * n_eq + n_in :, dim + n_eq + 2 * n_in + dim_ :
                ] = np.multiply(np.diag(D_2_c)[:, None], C_i)

                rhs = np.zeros(kkt.shape[0])
                rhs[:dim] = -dl_dzhat[i]
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
                qp.settings.eps_abs = 1.0e-4
                qp.settings.max_iter = 10
                default_rho = 1.0e-3
                qp.settings.default_rho = default_rho
                qp.settings.refactor_rho_threshold = default_rho
                qp.init(
                    H,
                    g,
                    spa.csc_matrix(kkt),
                    rhs,
                    C,
                    l,
                    u,
                )

            proxsuite.proxqp.sparse.solve_in_parallel(
                num_threads=ctx.cpu, qps=vector_of_qps
            )

            for i in range(nBatch):
                dx[i] = torch.from_numpy(
                    np.float64(vector_of_qps.get(i).results.x[:dim])
                )
                if n_eq > 0:
                    dnu[i] = torch.from_numpy(
                        np.float64(vector_of_qps.get(i).results.x[dim : dim + n_eq])
                    )
                dlam[i] = torch.from_numpy(
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
                bger(dlam.double(), zhats.double())
                + bger(ctx.lams.double(), dx.double())
                + bger(P_2_c_s_i.double(), b_6.double())
            )
            if G_e:
                dGs = dGs.mean(0)
            dhs = -dlam
            if h_e:
                dhs = dhs.mean(0)
            if neq > 0:
                dAs = (
                    bger(dnu.double(), zhats.double())
                    + bger(ctx.nus.double(), dx.double())
                    + bger(s_e.double(), b_5.double())
                )
                dbs = -dnu
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

            grads = (dQs, dps, dGs, dhs, dAs, dbs)

            return grads

    if structual_feasibility:
        return QPFunctionFn.apply
    else:
        return QPFunctionFn_infeas.apply
