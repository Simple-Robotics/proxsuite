# adapted from https://github.com/locuslab/optnet/blob/master/sudoku/train.py
import os
import time
import argparse
import numpy as np
import scipy.sparse as spa

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.autograd import Variable
    from torch.nn.parameter import Parameter

    import cvxpy as cp
    from proxsuite.torch.qplayer import QPFunction
except ImportError:
    print("Exiting script because torch is not installed.")
    exit(0)


def get_sudoku_matrix(n):
    X = np.array([[cp.Variable(n**2) for i in range(n**2)] for j in range(n**2)])
    cons = (
        [x >= 0 for row in X for x in row]
        + [cp.sum(x) == 1 for row in X for x in row]
        + [sum(row) == np.ones(n**2) for row in X]
        + [sum([row[i] for row in X]) == np.ones(n**2) for i in range(n**2)]
        + [
            sum([sum(row[i : i + n]) for row in X[j : j + n]]) == np.ones(n**2)
            for i in range(0, n**2, n)
            for j in range(0, n**2, n)
        ]
    )
    f = sum([cp.sum(x) for row in X for x in row])
    prob = cp.Problem(cp.Minimize(f), cons)

    A = np.asarray(prob.get_problem_data(cp.ECOS)[0]["A"].todense())
    A0 = [A[0]]
    rank = 1
    for i in range(1, A.shape[0]):
        if np.linalg.matrix_rank(A0 + [A[i]], tol=1e-12) > rank:
            A0.append(A[i])
            rank += 1

    return np.array(A0)


class QPLayer(nn.Module):
    def __init__(self, n, omp_parallel=False, maxIter=1000):
        super().__init__()
        self.maxIter = maxIter
        self.omp_parallel = omp_parallel
        nx = (n**2) ** 3
        self.Q = Variable(torch.zeros(nx, nx).double())
        self.G = Variable(-torch.eye(nx).double())
        self.u = Variable(torch.zeros(nx).double())
        self.l = Variable(-1.0e20 * torch.ones(nx).double())
        t = get_sudoku_matrix(n)
        self.A = Parameter(torch.rand(t.shape).double())
        self.log_z0 = Parameter(torch.zeros(nx).double())

    def forward(self, puzzles):
        nBatch = puzzles.size(0)
        p = -puzzles.view(nBatch, -1)
        b = self.A.mv(self.log_z0.exp())
        x, _, _ = QPFunction(maxIter=self.maxIter, omp_parallel=self.omp_parallel)(
            self.Q, p.double(), self.A, b, self.G, self.l, self.u
        )
        return x.float().view_as(puzzles)


class QPlayer_Learn_feasibility(nn.Module):
    def __init__(self, n, omp_parallel=False, maxIter=1000):
        super().__init__()
        self.maxIter = maxIter
        self.omp_parallel = omp_parallel

        nx = (n**2) ** 3
        Qpenalty = 0.0
        self.Q = Variable(Qpenalty * torch.eye(nx).double())

        self.G = Variable(-torch.eye(nx).double())
        self.h = Variable(torch.zeros(nx).double())
        self.l = Variable(-1.0e20 * torch.ones(nx).double())
        t = get_sudoku_matrix(n)
        self.A = Parameter(torch.rand(t.shape).double())
        self.b = Variable(torch.ones(self.A.size(0)).double())

    def forward(self, puzzles):
        nBatch = puzzles.size(0)

        p = -puzzles.view(nBatch, -1)

        x, y, z, s_e, s_i = QPFunction(
            structural_feasibility=False, omp_parallel=self.omp_parallel
        )(
            self.Q, p.double(), self.A, self.b, self.G, self.l, self.h
        )  # s0 should converge towards zero
        return x.float().view_as(puzzles), s_e, s_i


def train(args, epoch, model, trainX, trainY, optimizer):
    batchSz = args.batchSz

    batch_data_t = torch.FloatTensor(
        batchSz, trainX.size(1), trainX.size(2), trainX.size(3)
    )
    batch_targets_t = torch.FloatTensor(
        batchSz, trainY.size(1), trainX.size(2), trainX.size(3)
    )

    batch_data = Variable(batch_data_t, requires_grad=False)
    batch_targets = Variable(batch_targets_t, requires_grad=False)
    for i in range(0, trainX.size(0), batchSz):
        start = time.time()
        batch_data.data[:] = trainX[i : i + batchSz]
        batch_targets.data[:] = trainY[i : i + batchSz]

        optimizer.zero_grad()
        preds = None
        s_e = None
        s_i = None
        if args.structural_feasibility:
            preds = model(batch_data)
        else:
            preds, s_e, s_i = model(batch_data)
        loss = nn.MSELoss()(preds, batch_targets)
        if not (args.structural_feasibility):
            loss += args.penalty * (s_e.norm(2) + s_i.norm(2))
        loss.backward()
        optimizer.step()
        err = computeErr(preds.data) / batchSz
        print(
            f"Epoch: {epoch} [{i+batchSz}/{trainX.size(0)} ({float(i+batchSz)/trainX.size(0)*100:.0f}%)]\tLoss: {loss.item():.4f} Err: {err:.4f} Time: {time.time()-start:.2f}s"
        )


def test(args, epoch, model, testX, testY):
    batchSz = args.testBatchSz

    test_loss = 0
    batch_data_t = torch.FloatTensor(
        batchSz, testX.size(1), testX.size(2), testX.size(3)
    )
    batch_targets_t = torch.FloatTensor(
        batchSz, testY.size(1), testX.size(2), testX.size(3)
    )
    batch_data = Variable(batch_data_t)
    batch_targets = Variable(batch_targets_t)

    nErr = 0
    for i in range(0, testX.size(0), batchSz):
        print("Testing model: {}/{}".format(i, testX.size(0)), end="\r")
        with torch.no_grad():
            batch_data.data[:] = testX[i : i + batchSz]
            batch_targets.data[:] = testY[i : i + batchSz]
            output = None
            if args.structural_feasibility:
                output = model(batch_data)
            else:
                output, _, _ = model(batch_data)
            test_loss += nn.MSELoss()(output, batch_targets)
            nErr += computeErr(output.data)

    nBatches = testX.size(0) / batchSz
    test_loss = test_loss.item() / nBatches
    test_err = nErr / testX.size(0)
    print("TEST SET RESULTS:" + " " * 20)
    print(f"Average loss: {test_loss:.4f}")
    print(f"Err: {test_err:.4f}")


def computeErr(pred):
    batchSz = pred.size(0)
    nsq = int(pred.size(1))
    n = int(np.sqrt(nsq))
    s = (nsq - 1) * nsq // 2  # 0 + 1 + ... + n^2-1
    I = torch.max(pred, 3)[1].squeeze().view(batchSz, nsq, nsq)

    def invalidGroups(x):
        valid = x.min(1)[0] == 0
        valid *= x.max(1)[0] == nsq - 1
        valid *= x.sum(1) == s
        return ~valid

    boardCorrect = torch.ones(batchSz).type_as(pred)
    for j in range(nsq):
        # Check the jth row and column.
        boardCorrect[invalidGroups(I[:, j, :])] = 0
        boardCorrect[invalidGroups(I[:, :, j])] = 0

        # Check the jth block.
        row, col = n * (j // n), n * (j % n)
        M = invalidGroups(
            I[:, row : row + n, col : col + n].contiguous().view(batchSz, -1)
        )
        boardCorrect[M] = 0

        if boardCorrect.sum() == 0:
            return batchSz

    return batchSz - boardCorrect.sum().item()


if __name__ == "__main__":
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSz", type=int, default=150)
    parser.add_argument("--testBatchSz", type=int, default=200)
    parser.add_argument("--nEpoch", type=int, default=1)
    parser.add_argument("--penalty", type=float, default=0.001)
    parser.add_argument("--structural_feasibility", type=bool, default=True)
    parser.add_argument("--testPct", type=float, default=0.1)
    parser.add_argument("--omp_parallel", type=bool, default=False)
    args = parser.parse_args()

    # load dataset created with https://github.com/locuslab/optnet/blob/master/sudoku/create.py
    # default board size is 2
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    with open(f"{cur_dir}/data/features.pt", "rb") as f:
        X = torch.load(f)
    with open(f"{cur_dir}/data/labels.pt", "rb") as f:
        Y = torch.load(f)

    N, nFeatures = X.size(0), int(np.prod(X.size()[1:]))

    nTrain = int(N * (1.0 - args.testPct))
    nTest = N - nTrain

    trainX = X[:nTrain]
    trainY = Y[:nTrain]
    testX = X[nTrain:]
    testY = Y[nTrain:]

    assert nTrain % args.batchSz == 0
    assert nTest % args.testBatchSz == 0
    model = None
    # we try to learn the equality constraint matrix of the Sudoku problem
    if args.structural_feasibility:
        # the layer is during the whole training structurally feasible
        model = QPLayer(n=2, omp_parallel=args.omp_parallel)
    else:
        # the layer is not structurally feasible (i.e., during training the QP can be infeasible)
        # Nevertheless QPLayer will drive towards feasibility at test time the layer.
        # This learning process is more structured, since the layer will more comply to Sudoku rules
        # (indeed, for a Sudoku problem, the vector of ones must lie in the range space of the equality matrix).
        # It results with a harder problem to solve, but a quicker learning procedure
        # (i.e., less epoch are needed to have no prediction error).
        model = QPlayer_Learn_feasibility(n=2, omp_parallel=args.omp_parallel)
    lr = 5.0e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)

    test(args, 0, model, testX, testY)
    for epoch in range(1, args.nEpoch + 1):
        train(args, epoch, model, trainX, trainY, optimizer)
        test(args, epoch, model, testX, testY)
