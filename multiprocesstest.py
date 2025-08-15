from MatSamp import *
import matplotlib.pyplot as plt
import igraph as ig
import networkx as nx
import pickle
import multiprocessing as mp
import copy
from scipy.integrate import cumulative_trapezoid
import time
import os
from numpy.random import default_rng

# from multiprocessing import Process, Queue
import random

# print(mp.cpu_count())
plt.rcParams.update(
    {"text.usetex": True, "font.family": "STIXGeneral", "mathtext.fontset": "stix"}
)

track_sampler = []


def updater(q, S, dS, area, log_p, H, sampler_num, delta):

    S_view = np.frombuffer(S.get_obj(), dtype=np.float64)
    dS_view = np.frombuffer(dS.get_obj(), dtype=np.float64)
    # area = np.zeros(len(bin_edges) - 1)
    # log_p = np.zeros(len(bin_edges) - 1)
    area_view = np.frombuffer(area.get_obj(), dtype=np.float64)
    log_p_view = np.frombuffer(log_p.get_obj(), dtype=np.float64)
    H_view = np.frombuffer(H.get_obj(), dtype=np.int32)
    f = 1e-2
    ps_cnt, ns_cnt = 0, 0
    samples_nr = 0
    samples = 0
    finished = 0
    ps_min, ns_min = 1e9, 1e9

    while finished < sampler_num:
        data = q.get()

        if data is None:
            finished += 1
        else:
            idx, area_data, logp_data, ps_data, ns_data = data
            H_view[idx] += 1
            area_view[idx] += area_data
            log_p_view[idx] += logp_data
            samples += 1
            samples_nr += 1
            ps_cnt, ns_cnt = ps_cnt + int(ps_data == 1), ns_cnt + int(ns_data == 1)
            ps_min, ns_min = min(ps_min, ps_data), min(ns_min, ns_data)
            if ps_cnt >= sampler_num and ns_cnt >= sampler_num:
                ps_cnt, ns_cnt = 0, 0
                f = f / 2
                print("step size decreased, f={:.3f}".format(f))
            S_view[idx] += dS_view[idx] * f / delta
            # print("write", S_view[idx], end=" ", flush=True)

        # if samples % 1000 == 0:

        if samples >= 1e3:
            print(
                "{:d}:({:d},{:d})".format(samples_nr, ps_min, ns_min),
                end=" ",
                flush=True,
            )
            nz_idx = (H_view > 0) & (area_view != 0)
            dS_view[nz_idx] = cumulative_trapezoid(
                log_p_view[nz_idx] / area_view[nz_idx],
                bin_centers[nz_idx],
                initial=0,
            )
            dS_view[nz_idx] = dS_view[nz_idx] - min(dS_view[nz_idx]) + 1
            samples = 0


def sampler(q, pid, SNet, iter_cnt, bin_edges, bins_center, S):
    rng = default_rng()
    while iter_cnt:

        pscur, nscur = SNet.count_checker()
        rcur = SNet.assortativity_coeff()

        if pscur <= 1:  # no pos swt
            rho_cur = 0
            posswt = False
        elif nscur <= 1:  # no neg swt
            rho_cur = 1
            posswt = True
        else:
            rho_cur = 0.5#(rcur + 1) / 2
            p_posswt = rng.random()
            posswt = p_posswt < rho_cur

        swt = find_random_checker(
            SNet.A, SNet.pc, SNet.nc, SNet.pc_rows, SNet.nc_rows, posswt
        )
        i, j, k, l = swt

        area = (
            (SNet.deg[i] - SNet.deg[j])
            * (SNet.deg[k] - SNet.deg[l])
            / (SNet.M3 - SNet.M1)
        )

        # if area == 0:
        #     continue

        SNet.switch(swt)
        rnxt = SNet.assortativity_coeff()
        psnxt, nsnxt = SNet.count_checker()
        ps, ns = psnxt, nsnxt
        rho_nxt = (rnxt + 1) / 2

        curidx = np.searchsorted(bins_center, rcur, side="right") - 1
        nxtidx = np.searchsorted(bins_center, rnxt, side="right") - 1

        S_view = np.frombuffer(S.get_obj(), dtype=np.float64)

        assert any(S_view) != np.inf
        # print(len(S_view), curidx, nxtidx)
        # print("read", end=" ", flush=True)
        Scur = S_view[curidx] + (S_view[curidx + 1] - S_view[curidx]) * (
            rcur - bins_center[curidx]
        ) / (bins_center[curidx + 1] - bins_center[curidx])

        Snxt = S_view[nxtidx] + (S_view[nxtidx + 1] - S_view[nxtidx]) * (
            rnxt - bins_center[nxtidx]
        ) / (bins_center[nxtidx + 1] - bins_center[nxtidx])

        gcur = np.log((rho_cur / pscur if posswt else (1 - rho_cur) / nscur))
        gnxt = np.log(((1 - rho_nxt) / nsnxt if posswt else rho_nxt / psnxt))
        # print(
        #     "{:2d}: {:10.5f}->{:10.5f}{:15.2f} {:15.2f} {:5.2f} {:5.2f} {:15.2f} ({:6d}-{:6d}) ({:6d}-{:6d}) {:5.3f}".format(
        #         pid,
        #         rcur,
        #         rnxt,
        #         Scur,
        #         Snxt,
        #         gnxt,
        #         gcur,
        #         Scur - Snxt + gnxt - gcur,
        #         pscur,
        #         nscur,
        #         psnxt,
        #         nsnxt,
        #         p_posswt,
        #     ),
        #     posswt,
        #     end=" ",
        # )
        # switch back
        if (
            (np.log(np.random.rand()) > Scur - Snxt + gnxt - gcur)
            or psnxt == 0
            or nsnxt == 0
        ):
            # print("rejected", end=" ")
            SNet.switch(swt)
            ps, ns = pscur, nscur
        # print()
        idx = np.searchsorted(bin_edges, SNet.assortativity_coeff(), side="right") - 1
        iter_cnt -= 1
        q.put((idx, area, np.log(ps / ns), ps, ns))

    q.put(None)


if __name__ == "__main__":

    # state = np.random.get_state()
    random.seed(1)
    np.random.seed(1)
    n = 32
    p = np.round(1.2 * np.log(n) / n, 2)
    net = ig.Graph.Erdos_Renyi(n=n, p=p)

    A = ig_to_A(net)
    SNet = MatSamp(A, False)
    step = 2 * (SNet.M3 - SNet.M1)

    bin_cnt = int(min(1000, step))
    print(bin_cnt)
    bin_edges = np.linspace(-1, 1, bin_cnt + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_delta = bin_centers[1] - bin_centers[0]

    sample_num = 1e6
    sampler_num = 50
    S = mp.Array("d", bin_cnt)
    dS = mp.Array("d", bin_cnt)
    area = mp.Array("d", bin_cnt)
    log_p = mp.Array("d", bin_cnt)
    H = mp.Array("i", bin_cnt)
    S_view = np.frombuffer(S.get_obj(), dtype=np.float64)
    S_view[:] = 0
    dS_view = np.frombuffer(dS.get_obj(), dtype=np.float64)
    dS_view[:] = 1

    q = mp.Queue(maxsize=sampler_num)

    updater = mp.Process(
        target=updater, args=(q, S, dS, area, log_p, H, sampler_num, bin_delta)
    )
    updater.start()

    samplers = []
    for i in range(sampler_num):
        SNet = MatSamp(A, False)
        SNet.trackCheckerboard = True
        samplers.append(
            mp.Process(
                target=sampler,
                args=(
                    q,
                    i,
                    SNet,
                    int(sample_num / sampler_num),
                    bin_edges,
                    bin_centers,
                    S,
                ),
            )
        )

    start = time.time()
    for p in samplers:
        p.start()

    for p in samplers:
        p.join()
    updater.join()

    print("time:{:f}".format(time.time() - start))
    S = np.frombuffer(S.get_obj(), dtype=np.float64)
    H = np.frombuffer(H.get_obj(), dtype=np.int32)
    area = np.frombuffer(area.get_obj(), dtype=np.float64)
    log_p = np.frombuffer(log_p.get_obj(), dtype=np.float64)
    fix, ax = plt.subplots(4, 1, figsize=(10, 10))
    ax[0].plot(bin_centers[H != 0], H[H != 0])
    ax[1].plot(bin_centers[H != 0], S[H != 0])
    ax[2].plot(bin_centers[H != 0], area[H != 0] / H[H != 0])
    ax[3].plot(bin_centers[H != 0], log_p[H != 0] / H[H != 0])
    plt.tight_layout()
    plt.savefig("Experiments/mulproc.pdf", dpi=300)
