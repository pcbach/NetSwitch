import numpy as np


class NetSwitch:

    def __init__(self, G):
        self.A = np.array(G.get_adjacency().data, dtype=np.int8)
        self.n = np.shape(self.A)[0]
        self.deg = self.degree_seq()
        self.sort_adj()
        self.countonce = True
        self.checkercount_matrix()
        self.swt_done = 0

    def sort_adj(self):
        sortIdx = np.argsort(-self.deg)
        self.A = self.A[sortIdx, :][:, sortIdx]
        self.deg = self.deg[sortIdx]

    def checkercount_matrix(self):
        ''' Builds a matrix N, where element N[i,j] counts
        NEGATIVE checkerboards in row-pair (i, j) of the adjacency matrix A. '''

        self.N = np.zeros((self.n, self.n), dtype=np.int64)
        for i in range(self.n-1):
            for j in range(i+1, self.n):
                self.N[i, j] = self.count_rowpair_checkers_fast(i,j)
        self.Nrow = np.sum(self.N, axis=1)

    def count_rowpair_checkers(self, i, j):
        ''' For a pair (i, j) of rows in adjacency matrix A (i<j),
        checks all columns of A and counts all NEGATIVE checkerboards
        with coordinate (i, j, k, l).
        This is following the paper's implementation. '''

        r, s = 0, 0
        k_init = 0 if self.countonce is False else i+1
        for k in range(k_init, self.n):
            if k == i or k == j:
                continue
            if self.A[i, k] == 0 and self.A[j, k] == 1:
                r += 1
            if self.A[i, k] == 1 and self.A[j, k] == 0:
                s += r
        return s

    def count_rowpair_checkers_fast(self, i, j):
        ''' Alternative to self.count_rowpair_checkers(i, j)
        Here the main operation is vectorized with Numpy for speed.
        For a pair (i, j) of rows in adjacency matrix A (i<j),
        checks all columns of A and counts all NEGATIVE checkerboards
        with coordinate (i, j, k, l). '''

        all_checkerboard_sides = i+1 + np.nonzero(self.A[i, i+1:] ^ self.A[j, i+1:])[0]
        all_checkerboard_sides = np.delete(all_checkerboard_sides, np.where(all_checkerboard_sides == j))
        all_rightsides = np.nonzero(self.A[i, all_checkerboard_sides])[0]
        if all_rightsides.size==0:
            return int(0)
        else:
            cumsum_checkers = np.cumsum(np.diff(all_rightsides)-1)
            return int(all_rightsides[0]*(cumsum_checkers.size+1) + np.sum(cumsum_checkers))

    def update_N(self, swt):
        ''' Given a checkerboard (i, j, k, l) is switched,
        updates the matrix N that holds counts of checkerboards in matrix A '''

        i, j, k, l = swt
        for ref_row in [i, j, k, l]:
            for row in range(ref_row):
                self.N[row, ref_row] = self.count_rowpair_checkers_fast(row, ref_row)
                self.Nrow[row] = np.sum(self.N[row,:])
            for row in range(ref_row+1, self.n):
                self.N[ref_row, row] = self.count_rowpair_checkers_fast(ref_row, row)
                self.Nrow[ref_row] = np.sum(self.N[ref_row,:])

    def total_checkers(self):
        """Returns the total number of checkerboards left in the adjacency matrix"""
        return np.sum(self.Nrow)

    def switch(self, swt, update_N=True):
        """Switches a selected checkrboard in matrix A and calls for an update in checkerboard count
        given the coordinates (i, j, k, l), the checkerboars is at (i, k), (i, l), (j, k), (j, l)
        and the mirrored coordinates (k, i), (l, i), (k, j), (l, j) in matrix A"""
        i, j, k, l = swt
        self.A[i, k], self.A[i, l], self.A[j, k], self.A[j, l] =  1-self.A[i, k], 1-self.A[i, l], 1-self.A[j, k], 1-self.A[j, l]
        self.A[k, i], self.A[l, i], self.A[k, j], self.A[l, j] =  1-self.A[k, i], 1-self.A[l, i], 1-self.A[k, j], 1-self.A[l, j]
        self.update_N(swt)

    def find_random_checker(self, pos=True):
        if not pos:
            raise Exception("Finding random negative checkerboards is not implemented yet!!!")

        # FIND ROW I
        swt_idx = np.random.randint(np.sum(self.Nrow))+1
        Nrow_Cumsum = self.Nrow.cumsum()
        rnd_i = np.argwhere(Nrow_Cumsum>=swt_idx)[0][0]
        swt_idx -= 0 if (rnd_i == 0) else Nrow_Cumsum[rnd_i - 1]

        # FIND ROW J
        iRow_Cumsum = np.cumsum(self.N[rnd_i,rnd_i+1:])
        rnd_j = np.argwhere(iRow_Cumsum>=swt_idx)[0][0]
        swt_idx -= 0 if (rnd_j == 0) else iRow_Cumsum[rnd_j - 1]
        rnd_j += rnd_i + 1

        # FIND COLUMNS K & L
        all_checkerboard_sides = rnd_i+1 + np.nonzero(self.A[rnd_i, rnd_i+1:] ^ self.A[rnd_j, rnd_i+1:])[0]
        all_checkerboard_sides = np.delete(all_checkerboard_sides, np.where(all_checkerboard_sides == rnd_j))
        all_rightsides = np.nonzero(self.A[rnd_i, all_checkerboard_sides])[0]
        cumsum_checkers = np.cumsum(np.cumsum(np.insert(np.diff(all_rightsides)-1, 0, all_rightsides[0])))
        rnd_l = np.argwhere(cumsum_checkers>=swt_idx)[0][0]
        swt_idx -= 0 if (rnd_l == 0) else cumsum_checkers[rnd_l - 1]
        rnd_l = all_checkerboard_sides[all_rightsides[rnd_l]]
        rnd_k = all_checkerboard_sides[np.nonzero(S.A[rnd_j, all_checkerboard_sides])[0][swt_idx-1]]

        return (rnd_i, rnd_j, rnd_k, rnd_l)

    def largest_kl(self, row_i, row_j):
        for left_k in range(row_i+1, self.n-1):
            if self.A[row_i, left_k] == 0 and self.A[row_j, left_k] == 1:
                break
        for rght_l in range(self.n-1, left_k, -1):
            if rght_l == row_j:
                  continue
            if self.A[row_i, rght_l] == 1 and self.A[row_j, rght_l] == 0:
                break
        return left_k, rght_l

    def next_ij_rowrow(self):
        while self.N[self.i, self.j] == 0:
            self.j -= 1
            if self.j == self.i:
                self.i += 1
                self.j = self.n-1
                if self.i == self.n-1:
                    self.i = 0
        ord_k, ord_l = self.largest_kl(self.i, self.j)
        return (self.i, self.j, ord_k, ord_l)

    def next_ij_diag(self):
        row_dist = self.j - self.i
        while self.N[self.i, self.j] == 0:
            self.i += 1
            self.j += 1
            if self.j == self.n:
                row_dist -= 1
                self.i = 0
                self.j = self.i + row_dist
                if row_dist == 0:
                    row_dist, self.i, self.j = self.n-1, 0, self.n-1
        ord_k, ord_l = self.largest_kl(self.i, self.j)
        return (self.i, self.j, ord_k, ord_l)

    def switch_A(self, alg='RAND', count=-1):
        swt_num = 0
        if count == -1:
            count = self.total_checkers()
        while count > 0 and self.total_checkers()>0:
            match alg:
                case 'RAND':
                    swt = self.find_random_checker()
                case 'ORDR':
                    if self.swt_done == 0:
                        self.i, self.j = 0, self.n-1
                    swt = self.next_ij_rowrow()
                case 'ORDD':
                    if self.swt_done == 0:
                        self.i, self.j = 0, self.n-1
                    swt = self.next_ij_diag()
                case _:
                    raise Exception("No such switching algorithm!!!")

            i, j, k, l = swt
            #print([[self.A[i, k], self.A[i, l]], [self.A[j, k], self.A[j, l]]])
            self.switch(swt)
            self.swt_done += 1
            swt_num += 1
            count -= 1

        return swt_num if self.total_checkers() == 0 else -1

    def Havel_Hakimi(self, replace_adj=False):
        HH_adj = np.zeros((self.n, self.n))
        sorted_nodes = [i for i in zip(self.deg.copy(), range(self.n))]
        v1 = sorted_nodes[0]
        this_degree = v1[0]
        while this_degree > 0:
            if this_degree >= self.n:
                return False
            else:
                for v2_idx in range(1, this_degree+1):
                    v2 = sorted_nodes[v2_idx]
                    if v2[0] == 0:
                        return False
                    else:
                        sorted_nodes[v2_idx] = (v2[0]-1, v2[1])
                        HH_adj[v1[1], v2[1]], HH_adj[v2[1], v1[1]] = 1, 1
                sorted_nodes[0] = (0, sorted_nodes[0][1])
                sorted_nodes = sorted(sorted_nodes, key=lambda x: (x[0], -x[1]), reverse=True)
                v1 = sorted_nodes[0]
                this_degree = v1[0]
        if replace_adj:
            self.A = HH_adj
            return True
        else:
            return HH_adj

    def degree_seq(self):
      '''Returns the degree sequence of a graph from its adjacency matrix.'''
      return np.sum(self.A, axis=1)

    def assortativity_coeff(self):
        '''Calculates the assortativity coefficient for a graph
        from its binary adjacncy matrix.
        Calculations based on [PHYSICAL REVIEW E 84, 047101 (2011)].'''
        m = np.sum(self.A)/2.0
        all_i, all_j = np.where(np.triu(self.A))
        M2 = np.sum(self.deg[all_i] * self.deg[all_j]) / m
        di1 = (np.sum(self.deg[all_i] + self.deg[all_j]) / (m * 2.0) ) ** 2
        di2 = np.sum(self.deg[all_i]**2 + self.deg[all_j]**2) / (m * 2.0)
        return (M2 - di1)/(di2 - di1)
