import numpy as np
from scipy.sparse.linalg import eigsh
from sympy.strategies.core import switch
import itertools
import random
import igraph as ig

from numba import jit, int64, float32, types
from numba.experimental import jitclass

spec = [
    ("A", int64[:, :]),
    ("A2", int64[:, :]),
    ("A3", int64[:, :]),
    ("n", int64),
    ("deg", int64[:]),
    ("M1", float32),
    ("M3", float32),
    ("M2", float32),
    ("triad2", int64),
    ("triad3", int64),
    ("membership", int64[:]),
    ("pc", int64[:, :]),
    ("nc", int64[:, :]),
    ("pc_rows", int64[:]),
    ("nc_rows", int64[:]),
    ("bipartite", types.boolean),
    ("trackMotif", types.boolean),
    ("trackCheckerboard", types.boolean),
]


@jit(nopython=True)
def assortativity_coeff(A, deg, zagreb=False):
    """Calculates the assortativity coefficient for a graph
    from its binary adjacncy matrix.
    Calculations based on [PHYSICAL REVIEW E 84, 047101 (2011)]."""
    m = np.sum(A) / 2.0
    all_i, all_j = np.where(np.triu(A))
    M2 = np.sum(deg[all_i] * deg[all_j])
    di1 = (np.sum(deg[all_i] + deg[all_j]) / (m * 2.0)) ** 2 * m
    di2 = np.sum(deg[all_i] ** 2 + deg[all_j] ** 2) / (2.0)
    if not zagreb:
        return (M2 - di1) / (
            di2 - di1
        ), -1  # if (not zagreb) else ((M2 - di1) / (di2 - di1), M2)
    else:
        return (M2 - di1) / (di2 - di1), M2


@jit(nopython=True)
def count_rowpair_checkers_fast_upperswt(A, i, j):
    """Similar to self.count_rowpair_checkers_fast(i, j)
    but only counting the switchings with k, l > i, j
    i.e., upper trianlge checkerboards"""
    if j < i:
        i, j = j, i
    all_checkerboard_sides = j + 1 + np.nonzero(A[i, j + 1 :] ^ A[j, j + 1 :])[0]
    all_checkerboard_sides = remove_value(all_checkerboard_sides, j)

    all_rightsides_pos = np.nonzero(A[i, all_checkerboard_sides])[0]

    if all_rightsides_pos.size == 0:
        pos_count = int64(0)
    else:
        cumsum_checkers = np.cumsum(np.diff(all_rightsides_pos) - 1)
        pos_count = np.int64(
            all_rightsides_pos[0] * (cumsum_checkers.size + 1) + np.sum(cumsum_checkers)
        )
    all_rightsides_neg = np.nonzero(1 - A[i, all_checkerboard_sides])[0]

    if all_rightsides_neg.size == 0:
        neg_count = int64(0)
    else:
        cumsum_checkers = np.cumsum(np.diff(all_rightsides_neg) - 1)
        neg_count = np.int64(
            all_rightsides_neg[0] * (cumsum_checkers.size + 1) + np.sum(cumsum_checkers)
        )
    if pos_count + neg_count != all_rightsides_neg.size * all_rightsides_pos.size:
        print("FUCKKKK")
    return pos_count, neg_count


@jit(nopython=True)
def remove_value(arr, val):
    mask = arr != val
    return arr[mask]


@jit(nopython=True)
def count_rowpair_checkers_fast(A, i, j):
    """Alternative to self.count_rowpair_checkers(i, j)
    Here the main operation is vectorized with Numpy for speed.
    For a pair (i, j) of rows in adjacency matrix A (i<j),
    checks all columns of A and counts all NEGATIVE checkerboards
    with coordinate (i, j, k, l)."""

    all_checkerboard_sides = i + 1 + np.nonzero(A[i, i + 1 :] ^ A[j, i + 1 :])[0]
    all_checkerboard_sides = remove_value(all_checkerboard_sides, j)
    all_rightsides_pos = np.nonzero(A[i, all_checkerboard_sides])[0]
    if all_rightsides_pos.size == 0:
        pos_count = np.int64(0)
    else:
        cumsum_checkers = np.cumsum(np.diff(all_rightsides_pos) - 1)
        pos_count = np.int64(
            all_rightsides_pos[0] * (cumsum_checkers.size + 1) + np.sum(cumsum_checkers)
        )

    all_rightsides_neg = np.nonzero(1 - A[i, all_checkerboard_sides])[0]
    if all_rightsides_neg.size == 0:
        neg_count = np.int64(0)
    else:
        cumsum_checkers = np.cumsum(np.diff(all_rightsides_neg) - 1)
        neg_count = np.int64(
            all_rightsides_neg[0] * (cumsum_checkers.size + 1) + np.sum(cumsum_checkers)
        )
    return np.int64(pos_count), np.int64(neg_count)


@jit(nopython=True)
def update_count_mat(
    A, swt, pc, pc_rows, nc=None, nc_rows=None, count_upper=True, count_lower=True
):
    """Given a checkerboard (i, j, k, l) is switched,
    updates the matrix N that holds counts of checkerboards in matrix A"""
    for ref_row in swt:
        for row in range(ref_row):

            if count_lower and not count_upper:
                checker_count = np.array(count_rowpair_checkers_fast(A, row, ref_row))
                checker_count[0] -= count_rowpair_checkers_fast_upperswt(
                    A,
                    row,
                    ref_row,
                )[0]
                checker_count[1] -= count_rowpair_checkers_fast_upperswt(
                    A, row, ref_row
                )[1]
            elif count_upper and not count_lower:
                checker_count = np.array(
                    count_rowpair_checkers_fast_upperswt(
                        A,
                        row,
                        ref_row,
                    )
                )
            else:
                checker_count = np.array(count_rowpair_checkers_fast(A, row, ref_row))

            pc[row, ref_row] = np.int64(checker_count[0])
            nc[row, ref_row] = np.int64(checker_count[1])
            pc_rows[row] = np.sum(pc[row, :])
            nc_rows[row] = np.sum(nc[row, :])
        for row in range(ref_row + 1, np.shape(A)[0]):
            if count_lower and not count_upper:
                checker_count = np.array(count_rowpair_checkers_fast(A, ref_row, row))
                checker_count[0] -= count_rowpair_checkers_fast_upperswt(
                    A,
                    ref_row,
                    row,
                )[0]
                checker_count[1] -= count_rowpair_checkers_fast_upperswt(
                    A, ref_row, row
                )[1]
            elif count_upper and not count_lower:
                checker_count = np.array(
                    count_rowpair_checkers_fast_upperswt(
                        A,
                        ref_row,
                        row,
                    )
                )
            else:
                checker_count = np.array(count_rowpair_checkers_fast(A, ref_row, row))

            pc[ref_row, row] = np.int64(checker_count[0])
            nc[ref_row, row] = np.int64(checker_count[1])
            pc_rows[row] = np.sum(pc[row, :])
            nc_rows[row] = np.sum(nc[row, :])
        pc_rows[ref_row] = np.sum(pc[ref_row, :])
        nc_rows[ref_row] = np.sum(nc[ref_row, :])
    return (
        pc.astype(np.int64),
        pc_rows.astype(np.int64),
        nc.astype(np.int64),
        nc_rows.astype(np.int64),
    )


def ig_to_A(g):
    return np.array(g.get_adjacency().data, dtype=np.int64)


@jit(nopython=True)
def get_M2(deg, A):
    n = deg.shape[0]
    s = 0.0
    for i in range(n):
        for j in range(n):
            s += deg[i] * A[i, j] * deg[j]
    return 0.5 * s


@jit(nopython=True)
def get_col(mask, idx, cnt):
    if idx > cnt:
        return -1
    s = 0
    for i in range(len(mask)):
        s += mask[i]
        if s == idx:
            return i
    return -1


def find_random_checker(A, pc, nc, pc_rows, nc_rows, pos):
    """Using the count matrices, draws a checkerboard at random
    :param pos: True (False) to find a negative (positive) checkerboard configuration at random
    :return: A list of four different nodes i, j, k, l determining the switch"""
    if pos:
        count_mat = pc
        count_mat_row = pc_rows
        shifter = 0
    elif not pos:
        count_mat = nc
        count_mat_row = nc_rows
        shifter = 1
    # FIND ROW I
    swt_idx = np.random.randint(np.sum(count_mat_row)) + 1
    Nrow_Cumsum = count_mat_row.cumsum()
    rnd_i = np.argwhere(Nrow_Cumsum >= swt_idx)[0][0]
    swt_idx -= 0 if (rnd_i == 0) else Nrow_Cumsum[rnd_i - 1]

    # FIND ROW J
    iRow_Cumsum = np.cumsum(count_mat[rnd_i, rnd_i + 1 :])
    rnd_j = np.argwhere(iRow_Cumsum >= swt_idx)[0][0]
    swt_idx -= 0 if (rnd_j == 0) else iRow_Cumsum[rnd_j - 1]
    rnd_j += rnd_i + 1

    # FIND COLUMNS K & L
    all_checkerboard_sides = (
        rnd_i + 1 + np.nonzero(A[rnd_i, rnd_i + 1 :] ^ A[rnd_j, rnd_i + 1 :])[0]
    )
    all_checkerboard_sides = np.delete(
        all_checkerboard_sides, np.where(all_checkerboard_sides == rnd_j)
    )
    all_rightsides = np.nonzero(A[rnd_i, all_checkerboard_sides] - shifter)[0]
    cumsum_checkers = np.cumsum(
        np.cumsum(np.insert(np.diff(all_rightsides) - 1, 0, all_rightsides[0]))
    )
    rnd_l = np.argwhere(cumsum_checkers >= swt_idx)[0][0]
    swt_idx -= 0 if (rnd_l == 0) else cumsum_checkers[rnd_l - 1]
    rnd_l = all_checkerboard_sides[all_rightsides[rnd_l]]
    rnd_k = all_checkerboard_sides[
        np.nonzero(A[rnd_j, all_checkerboard_sides] - shifter)[0][swt_idx - 1]
    ]

    return (rnd_i, rnd_j, rnd_k, rnd_l)


@jitclass(spec)
class MatSamp:

    def __init__(self, A, bipartite):
        self.A = A  # , dtype=np.int8
        self.n = np.shape(self.A)[0]
        self.deg = np.sum(self.A, axis=0).astype(np.int64)

        # assortativity constant
        self.M3 = 1 / 2 * np.sum(self.deg**3)
        self.M1 = np.sum(1 / 2 * self.deg**2) ** 2 / (np.sum(self.deg) / 2)
        self.M2 = get_M2(self.deg, self.A)

        # sort
        self.sort_adj()

        # bipartite
        self.membership = -np.ones(self.n).astype(np.int64)
        self.bipartite = False
        if bipartite:
            self.set_bipartite()

        # motif
        Af = self.A.astype(np.float32)
        A2f = Af @ Af
        A3f = A2f @ Af
        self.A2 = A2f.astype(np.int64)
        self.A3 = A3f.astype(np.int64)

        self.triad3 = np.int64(0)
        self.triad2 = np.int64(0)

        self.trackMotif = False
        self.trackCheckerboard = False

        for i in range(self.n):
            self.triad3 += self.A3[i, i]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.triad2 += self.A2[i, j]

        # checkerboard
        self.pc = np.zeros((self.n, self.n)).astype(np.int64)
        self.nc = np.zeros((self.n, self.n)).astype(np.int64)
        self.pc_rows = np.zeros(self.n).astype(np.int64)
        self.nc_rows = np.zeros(self.n).astype(np.int64)
        if self.bipartite:
            self.checkercount_matrix(count_lower=False, count_upper=True)
        else:
            self.checkercount_matrix()

    def set_bipartite(self):
        for v in range(self.n):
            if self.membership[v] == -1:
                queue = [v]
                self.membership[v] = 0  # Assign first part as False
                while queue:
                    u = queue.pop()
                    for w in range(self.n):
                        if self.A[w, u] == 0:
                            continue
                        if self.membership[w] == -1:
                            self.membership[w] = 1 - self.membership[u]
                            queue.append(w)
                        elif self.membership[w] == self.membership[u]:
                            print(self.membership)
                            print("Network not bipartite")
                            self.membership = np.zeros(self.n).astype(np.int64)
                            return

        # sortIdx = np.argsort(self.membership,kind = 'stable')
        # sortIdx = np.lexsort((self.membership,np.arange(self.n)))
        idx = 0
        sortIdx = np.zeros(self.n, dtype=np.int64)
        for i in range(self.n):
            if self.membership[i]:
                sortIdx[idx] = i
                idx += 1
        for i in range(self.n):
            if not self.membership[i]:
                sortIdx[idx] = i
                idx += 1

        self.A = self.A[sortIdx, :][:, sortIdx]
        self.deg = self.deg[sortIdx]
        self.membership = self.membership[sortIdx]
        self.bipartite = True

        return

    def sort_adj(self):
        sortIdx = np.argsort(-self.deg)
        self.A = self.A[sortIdx, :][:, sortIdx]
        self.deg = self.deg[sortIdx]

    def assortativity_coeff(self):
        if (self.M3 - self.M1) == 0:
            return 0
        return (self.M2 - self.M1) / (self.M3 - self.M1)

    def count_checker(self):
        """Returns the total number of checkerboards left in the adjacency matrix"""
        if not self.trackCheckerboard:
            if self.bipartite:
                self.checkercount_matrix(count_lower=False, count_upper=True)
            else:
                self.checkercount_matrix()
        return np.sum(self.pc_rows), np.sum(self.nc_rows)

    def get_random_row_pair(self):
        r = np.random.randint(0, self.n * (self.n - 1) // 2 - 1)

        i = 0
        while r >= self.n - i - 1:
            r -= self.n - i - 1
            i += 1
        j = i + 1 + r
        return i, j

    def M2_after_swt(self, swt):
        i, j, k, l = swt

        if i == -1 or j == -1 or k == -1 or l == -1:
            return self.M2

        return self.M2 + (
            self.deg[i] * self.deg[k] * (1 - 2 * self.A[i, k])
            + self.deg[j] * self.deg[l] * (1 - 2 * self.A[j, l])
            + self.deg[l] * self.deg[i] * (1 - 2 * self.A[l, i])
            + self.deg[k] * self.deg[j] * (1 - 2 * self.A[k, j])
        )

    def assort_after_swt(self, swt):

        if (self.M3 - self.M1) == 0:
            return 0
        M2_after = self.M2_after_swt(swt)

        return (M2_after - self.M1) / (self.M3 - self.M1)

    def switch(self, swt):
        i, j, k, l = swt

        if i == -1 or j == -1 or k == -1 or l == -1:
            return

        self.M2 += (
            self.deg[i] * self.deg[k] * (1 - 2 * self.A[i, k])
            + self.deg[j] * self.deg[l] * (1 - 2 * self.A[j, l])
            + self.deg[l] * self.deg[i] * (1 - 2 * self.A[l, i])
            + self.deg[k] * self.deg[j] * (1 - 2 * self.A[k, j])
        )

        if self.trackMotif:
            self.update_motif(swt)

        self.A[i, k], self.A[i, l], self.A[j, k], self.A[j, l] = (
            1 - self.A[i, k],
            1 - self.A[i, l],
            1 - self.A[j, k],
            1 - self.A[j, l],
        )
        self.A[k, i], self.A[l, i], self.A[k, j], self.A[l, j] = (
            1 - self.A[k, i],
            1 - self.A[l, i],
            1 - self.A[k, j],
            1 - self.A[l, j],
        )

        if self.trackCheckerboard:
            if self.bipartite:
                self.update_count_mat(swt, count_lower=False, count_upper=True)
            else:
                self.update_count_mat(swt)

    def next(self, mode):
        i, j = self.get_random_row_pair()

        if self.bipartite and self.membership[i] != self.membership[j]:
            return (-1, -1, -1, -1)

        iAllNb = self.A[i, :]
        jAllNb = self.A[j, :]
        mask = iAllNb ^ jAllNb
        iNb = mask & iAllNb
        jNb = mask & jAllNb

        iNb[j] = 0
        jNb[i] = 0

        if self.bipartite:
            memMask = (
                self.membership[i] + (1 - 2 * self.membership[i]) * self.membership
            ).astype(
                np.int64
            )  # 1 if != M[i], 0 if == M[i]
            iNb = iNb & memMask
            jNb = jNb & memMask

        iCnt = np.sum(iNb)
        jCnt = np.sum(jNb)

        if iCnt >= 1 and jCnt >= 1:
            rk, rl = random.randint(1, iCnt), random.randint(1, jCnt)
            k, l = get_col(iNb, rk, iCnt), get_col(jNb, rl, jCnt)
            if mode == "rand":
                return (i, j, k, l)
            elif mode == "pos" and k > l:
                return (i, j, k, l)
            elif mode == "neg" and k < l:
                return (i, j, k, l)

        return (-1, -1, -1, -1)

    def get_swt_prob(self, swt):
        i, j, k, l = swt

        if self.bipartite and self.membership[i] != self.membership[j]:
            return 0

        iAllNb = self.A[i, :]
        jAllNb = self.A[j, :]
        mask = iAllNb ^ jAllNb
        iNb = mask & iAllNb
        jNb = mask & jAllNb

        iNb[j] = 0
        jNb[i] = 0

        if self.bipartite:
            memMask = (
                self.membership[i] + (1 - 2 * self.membership[i]) * self.membership
            ).astype(
                np.int64
            )  # 1 if != M[i], 0 if == M[i]
            iNb = iNb & memMask
            jNb = jNb & memMask

        iCnt = np.sum(iNb)
        jCnt = np.sum(jNb)

        return 2 / self.n / (self.n - 1) / iCnt / jCnt

    def update_motif(self, swt):
        i, j, k, l = swt

        if self.trackMotif:
            self.update_motif_edge(i, k)
            self.update_motif_edge(j, l)
            self.update_motif_edge(l, i)
            self.update_motif_edge(k, j)

            self.update_motif_edge(k, i)
            self.update_motif_edge(l, j)
            self.update_motif_edge(i, l)
            self.update_motif_edge(j, k)

        self.A[i, k], self.A[i, l], self.A[j, k], self.A[j, l] = (
            1 - self.A[i, k],
            1 - self.A[i, l],
            1 - self.A[j, k],
            1 - self.A[j, l],
        )
        self.A[k, i], self.A[l, i], self.A[k, j], self.A[l, j] = (
            1 - self.A[k, i],
            1 - self.A[l, i],
            1 - self.A[k, j],
            1 - self.A[l, j],
        )

    def update_motif_edge(self, i, j):
        delta = 1 - 2 * self.A[i, j]  # 1 add edge, -1 remove edge

        for k in range(self.n):
            self.A2[k, j] += delta * self.A[k, i]
            self.A2[i, k] += delta * self.A[j, k]

            self.triad2 += delta * (self.A[j, k] * (k > i) + self.A[k, i] * (j > k))

        for k in range(self.n):

            self.A3[k, j] += delta * self.A2[k, i]
            self.A3[i, k] += delta * self.A2[j, k]

            self.triad3 += delta * (self.A2[k, i] * (j == k) + self.A2[j, k] * (i == k))

            for l in range(self.n):
                self.A3[k, l] += delta * self.A[k, i] * self.A[j, l]

                self.triad3 += delta * self.A[k, i] * self.A[j, l] * (l == k)

        self.A3[i, j] -= self.A[j, i]
        self.A[i, j] = 1 - self.A[i, j]

    def count_motif(self):
        if not self.trackMotif:
            Af = self.A.astype(np.float64)
            A2 = Af @ Af
            A3 = A2 @ Af

            triad_3 = 0
            for i in range(self.n):
                triad_3 += A3[i, i]
            self.triad3 = np.int64(triad_3)

            triad_2 = 0
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    triad_2 += A2[i, j]
            self.triad2 = np.int64(triad_2)
            return (self.triad2 - self.triad3 // 2, self.triad3 // 6)

        return (self.triad2 - self.triad3 // 2, self.triad3 // 6)

    def checkercount_matrix(self, count_upper=True, count_lower=True):
        """Builds a matrix N, where element N[i,j] counts
        NEGATIVE checkerboards in row-pair (i, j) of the adjacency matrix A."""

        # self.pc = np.zeros((self.n, self.n), dtype=np.int64)
        # self.nc = np.zeros((self.n, self.n), dtype=np.int64)

        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                if count_lower and not count_upper:
                    self.pc[i, j], self.nc[i, j] = np.array(
                        count_rowpair_checkers_fast(self.A, i, j)
                    ) - np.array(
                        (
                            (np.int64(0), np.int64(0))
                            if count_upper
                            else count_rowpair_checkers_fast_upperswt(self.A, i, j)
                        )
                    )
                elif count_upper and not count_lower:
                    self.pc[i, j], self.nc[i, j] = np.array(
                        count_rowpair_checkers_fast_upperswt(self.A, i, j)
                    )
                else:
                    self.pc[i, j], self.nc[i, j] = np.array(
                        count_rowpair_checkers_fast(self.A, i, j)
                    )
        self.pc_rows = np.sum(self.pc, axis=1).astype(np.int64)
        self.nc_rows = np.sum(self.nc, axis=1).astype(np.int64)

    def update_count_mat(self, swt, count_upper=True, count_lower=True):
        """Given a checkerboard (i, j, k, l) is switched,
        updates the matrix N that holds counts of checkerboards in matrix A"""
        self.pc, self.pc_rows, self.nc, self.nc_rows = update_count_mat(
            self.A,
            swt,
            self.pc,
            self.pc_rows,
            self.nc,
            self.nc_rows,
            count_upper=count_upper,
            count_lower=count_lower,
        )


# n1 = 32  # number of vertices in set A
# n2 = 32  # number of vertices in set B
# p = 0.3  # probability of edge creation between parts
# # Create bipartite type list
# types = [0] * n1 + [1] * n2  # 0 = part A, 1 = part B
# # Generate all possible edges from A to B
# possible_edges = [(i, j + n1) for i in range(n1) for j in range(n2)]
# # Randomly select edges
# edges = [e for e in possible_edges if random.random() < p]
# # Create bipartite graph
# net = ig.Graph.Bipartite(types, edges)

# # import time

# # n = 16
# # net = ig.Graph.Erdos_Renyi(n=n, p=1.5 * np.log(n) / n)

# A = ig_to_A(net)
# SNet = MatSamp(A,True)
# SNet.trackCheckerboard = True
# SNet.trackMotif = True
# sample = []
# for i in range(10):
#     swt = SNet.next("rand")
#     if swt[0] != -1:
#         SNet.switch(swt)
#         print(SNet.count_checker())


# print(np.mean(np.diff(times)))

# SNet = MatSamp(A)
# sample = []

# times = [time.time()]
# for i in range(1000):
#     swt = SNet.next("rand")
#     SNet.total_checkers()
#     times.append(time.time())

# print(np.mean(np.diff(times)))
# print(SNet.A)
