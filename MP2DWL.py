from MatSamp import *
import matplotlib.pyplot as plt
import igraph as ig
import sys
import json
import multiprocessing as mp
import time
from numpy.random import default_rng
from collections import namedtuple
import os
import networkx as nx

# from multiprocessing import Process, Queue
import random

# print(mp.cpu_count())
plt.rcParams.update(
    {"text.usetex": True, "font.family": "STIXGeneral", "mathtext.fontset": "stix"}
)


graph_des = namedtuple("graph_des", ["type", "n", "k", "npseed", "seed"])
des = graph_des("None", "None", "None", "None", "None")
runid = 0
mean_sample_cnt = 0
folder_path = None
r_0 = None
q_0 = None
samples_0 = None


def bilinear_interpolation(
    left_col,
    right_col,
    upper_row,
    lower_row,
    value,
    mask,
    row,
    col,
):
    value = np.asarray(value, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if mask.any():
        mean_val = value[mask].min()
        value[~mask] = mean_val
        # mean_val = value[mask].mean()
        # value[~mask] = value[mask].max()
    upper_left_val, upper_right_val, lower_left_val, lower_right_val = value

    if upper_row == lower_row:
        mid_left_val = lower_left_val
    else:
        mid_left_val = lower_left_val + (upper_left_val - lower_left_val) * (
            row - lower_row
        ) / (upper_row - lower_row)

    if upper_row == lower_row:
        mid_right_val = lower_right_val
    else:
        mid_right_val = lower_right_val + (upper_right_val - lower_right_val) * (
            row - lower_row
        ) / (upper_row - lower_row)

    if right_col == left_col:
        mid_right_val = upper_right_val
        return mid_left_val
    else:
        return mid_left_val + (mid_right_val - mid_left_val) * (col - left_col) / (
            right_col - left_col
        )


# def centroid_interpolation(
#     top_left,top_right,bottom_left,bottom_right
# ):


# print(bilinear_interpolation(0,2,2,1,4,5,2,4,3,3))
# 0/0
def gaussian_kernel(size=3, sigma=1.0):
    ax = np.arange(-(size // 2), size // 2 + 1)
    X, Y = np.meshgrid(ax, ax)
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


k_size = 1
kernel = gaussian_kernel(size=k_size, sigma=0.5)
print(kernel)


def logsumlog(A):
    A_max = np.ma.max(A)
    exp_shifted = np.exp(A - A_max)
    s = np.sum(exp_shifted)
    if s <= 0 or not np.isfinite(s):
        return A_max
    return A_max + np.log(s)


def updater(
    q,
    S,
    H,
    sampler_num,
    delta_r,
    delta_q,
    terminate_flag,
    bin_cnt_r,
    bin_cnt_q,
):
    S_view = np.frombuffer(S.get_obj(), dtype=np.float64)
    H_view = np.frombuffer(H.get_obj(), dtype=np.int32)
    H_star = np.zeros(bin_cnt_q * bin_cnt_r)
    f = 2 ** (1)
    samples = samples_0 if samples_0 != None else 0
    finished = 0
    tunnelflag = np.zeros(sampler_num)
    min_q = 1e9
    max_q = -1e9
    tunnel = 0
    latest_tunnel = 0
    cover_H_view = np.sum((H_view != 0))
    cover_H_star = 0
    reset_check = 0
    while finished < sampler_num:
        data = q.get()

        if data is None:
            finished += 1
        else:
            pid, idx, idx_r, idx_q = data
            # if idx_q == 0:
            #    print(idx, idx_r, idx_q)
            if idx_q == 0 and tunnelflag[pid] != -1:
                tunnelflag[pid] = -1
                if latest_tunnel != -1:
                    tunnel += 0.5
                latest_tunnel = -1
            elif idx_q == bin_cnt_q - 1 and tunnelflag[pid] != 1:
                tunnelflag[pid] = 1
                if latest_tunnel != 1:
                    tunnel += 0.5
                latest_tunnel = 1

            min_q = min(min_q, idx_q)
            max_q = max(max_q, idx_q)

            samples += 1
            cover_H_view += int(H_view[idx] == 0)
            H_view[idx] += 1
            cover_H_star += int(H_view[idx] != 0 and H_star[idx] == 0)
            H_star[idx] += 1
            # mean_sample_cnt.value = samples / cover_H_view
            # if tunnel >= 1 and H_star_sum >= cover_H_view * 10:
            #     H_star = {}
            #     H_star_sum = 0
            #     H_star_min = 1e9
            #     H_star_cnt = 0

            # if not idx in H_star:
            #     H_star[idx] = 0
            #     H_star_cnt += 1#int(H_star[idx] == 0)
            # H_star_sum += 1
            # H_star_min = min(H_star.values())

            if k_size > 1:
                for dr in range(-(k_size // 2), (k_size // 2)):
                    for dq in range(-(k_size // 2), (k_size // 2)):
                        idx_r_, idx_q_ = idx_r + dr, idx_q + dq
                        idx_ = idx_r_ * bin_cnt_q + idx_q_
                        if (
                            0 <= idx_r_ < bin_cnt_r
                            and 0 <= idx_q_ < bin_cnt_q
                            and H_view[idx_] != 0
                        ):
                            S_view[idx_] += (
                                f * kernel[(k_size // 2) + dr, (k_size // 2) + dq]
                            )
            else:
                S_view[idx] += f

            # if idx // bin_cnt_q != idx_r or idx % bin_cnt_q != idx_q:
            #     print(idx // bin_cnt_q, idx % bin_cnt_q, idx_r, idx_q)
            #     raise "shit"
            # print(idx // bin_cnt_q, idx % bin_cnt_q, idx_r, idx_q)
            # S_view[idx] += f  # / delta_q / delta_r

            if (
                tunnel
                >= 5
                # and H_star_min / (H_star_sum / H_star_cnt) >= 0.8
                # and H_star_sum >= cover_H_view * 9
                # and samples > reset_check
                # and cover_H_star / cover_H_view > 0.9
            ):

                # while finished < sampler_num and not q.empty():
                #     data = q.get()
                #     if data is None:
                #         finished += 1
                #     else:
                #         pid, idx, idx_r, idx_q = data
                #         H_view[idx] += 1
                #         H_star[idx] += 1
                #         samples += 1
                #         if k_size > 1:
                #             for dr in range(-(k_size // 2), (k_size // 2)):
                #                 for dq in range(-(k_size // 2), (k_size // 2)):
                #                     idx_r_, idx_q_ = idx_r + dr, idx_q + dq
                #                     idx_ = idx_r_ * bin_cnt_q + idx_q_
                #                     if (
                #                         0 <= idx_r_ < bin_cnt_r
                #                         and 0 <= idx_q_ < bin_cnt_q
                #                         and H_view[idx_] != 0
                #                     ):
                #                         S_view[idx_] += (
                #                             f
                #                             * kernel[
                #                                 (k_size // 2) + dr, (k_size // 2) + dq
                #                             ]
                #                         )
                #         else:
                #             S_view[idx] += f

                tunnelflag = np.zeros(sampler_num)
                tunnel = 0
                latest_tunnel = 0
                H_star[:] = 0
                min_q = 1e9
                max_q = -1e9
                reset_check = samples + 1e6
                cover_H_star = 0

                f = f / 2

                print("{:d}: Step size decreased, f={:.3e}".format(samples, f))
                # print(folder_path)
                with open(folder_path + "/{:d}.json".format(runid), "w") as ffff:
                    json.dump(
                        {
                            "r0": r_0,
                            "q0": q_0,
                            "f": f,
                            "bin_cnt_r": bin_cnt_r,
                            "bin_cnt_q": bin_cnt_q,
                            "sample_num": samples,
                            "graph_description": des._asdict(),
                            "S": S_view.reshape(bin_cnt_r, bin_cnt_q).tolist(),
                            "H": H_view.reshape(bin_cnt_r, bin_cnt_q).tolist(),
                        },
                        ffff,
                        separators=(",", ":"),
                    )
                if f < 2 ** (-13):
                    terminate_flag.value = 0

        # print(samples, ": ", end="")
        if (
            samples % 1e3 == 0
        ):  # (samples < 1e7 and samples % 1e3 == 0) or (samples >= 1e7 and samples % 1e7 == 0):
            print(samples, ": ", end="")
            print(
                "{:.3e} {:.1f} {:.2f} {:.0f} {:.0f}".format(
                    f, tunnel, cover_H_star / cover_H_view, min_q, max_q
                ),
            )
            min_q = 1e9
            max_q = -1e9
        if samples % 1e5 == 0:
            # (
            #      (samples >= 1e7 and samples % 1e6 == 0)
            #      or samples == sampler_num
            #      or (samples < 1e7 and samples % 1e4 == 0)
            #  ):
            with open(folder_path + "/{:d}.json".format(runid), "w") as ffff:
                json.dump(
                    {
                        "r0": r_0,
                        "q0": q_0,
                        "f": f,
                        "bin_cnt_r": bin_cnt_r,
                        "bin_cnt_q": bin_cnt_q,
                        "sample_num": samples,
                        "graph_description": des._asdict(),
                        "S": S_view.reshape(bin_cnt_r, bin_cnt_q).tolist(),
                        "H": H_view.reshape(bin_cnt_r, bin_cnt_q).tolist(),
                    },
                    ffff,
                    separators=(",", ":"),
                )
            # print(samples, ": ", end="")
            # print(
            #     " ".join(
            #         "({:.1f} {:.0f} {:.0f})".format(tunnel_all[i], q_min[i], q_max[i])
            #         for i in range(int(sampler_num))
            #     )
            # )
            # print(
            #     samples, ":", np.sum(tunnel_all), np.min(tunnel_all), np.max(tunnel_all)
            # )


def sampler(
    q,
    pid,
    SNet,
    iter_cnt,
    bin_centers_r,
    bin_centers_q,
    S,
    H,
    terminate_flag,
):
    bin_cnt_r = len(bin_centers_r)
    bin_cnt_q = len(bin_centers_q)
    rng = default_rng()
    reject = 0
    S_view = np.frombuffer(S.get_obj(), dtype=np.float64)
    H_view = np.frombuffer(H.get_obj(), dtype=np.int32)

    while iter_cnt and terminate_flag.value:
        pscur, nscur = SNet.count_checker()
        ps, ns = pscur, nscur
        qcur = (nscur - pscur) / (nscur + pscur)
        rcur = SNet.assortativity_coeff()

        if pscur == 0:
            rho_cur = 0
            posswt = False
        elif nscur == 0:
            rho_cur = 1
            posswt = True
        else:
            rho_cur = 0.5
            # rho_cur = (kappa / 2 - qcur) / kappa
            p_posswt = rng.random()
            posswt = p_posswt < rho_cur

        swt = find_random_checker(
            SNet.A, SNet.pc, SNet.nc, SNet.pc_rows, SNet.nc_rows, posswt
        )

        SNet.switch(swt)
        rnxt = SNet.assortativity_coeff()
        psnxt, nsnxt = SNet.count_checker()
        qnxt = (nsnxt - psnxt) / (nsnxt + psnxt)
        ps, ns = psnxt, nsnxt

        # if abs(rnxt-qnxt) > 1:
        #    print(rnxt,ps,ns)
        if psnxt == 0:
            rho_nxt = 0
        elif nsnxt == 0:
            rho_nxt = 1
        else:
            rho_nxt = 0.5
            # rho_nxt = (kappa / 2 - qnxt) / kappa

        curidx_r = np.searchsorted(bin_centers_r, rcur, side="right") - 1
        nxtidx_r = np.searchsorted(bin_centers_r, rnxt, side="right") - 1

        curidx_q = np.searchsorted(bin_centers_q, qcur, side="right") - 1
        nxtidx_q = np.searchsorted(bin_centers_q, qnxt, side="right") - 1

        S_view = np.frombuffer(S.get_obj(), dtype=np.float64)
        H_view = np.frombuffer(H.get_obj(), dtype=np.int32)

        assert any(S_view) != np.inf

        curidx_q_ = curidx_q + 1 if curidx_q + 1 < bin_cnt_q else curidx_q - 1
        nxtidx_q_ = nxtidx_q + 1 if nxtidx_q + 1 < bin_cnt_q else nxtidx_q - 1
        curidx_r_ = curidx_r + 1 if curidx_r + 1 < bin_cnt_r else curidx_r - 1
        nxtidx_r_ = nxtidx_r + 1 if nxtidx_r + 1 < bin_cnt_r else nxtidx_r - 1

        # curidx_q_ = min(curidx_q + 1, bin_cnt_q - 1)
        # curidx_r_ = min(curidx_r + 1, bin_cnt_r - 1)
        # nxtidx_q_ = min(nxtidx_q + 1, bin_cnt_q - 1)
        # nxtidx_r_ = min(nxtidx_r + 1, bin_cnt_r - 1)

        corners_cur = [
            (curidx_r, curidx_q),  # upper left
            (curidx_r, curidx_q_),  # upper right
            (curidx_r_, curidx_q),  # lower left
            (curidx_r_, curidx_q_),  # lower right
        ]

        value_cur = [S_view[r * bin_cnt_q + q] for r, q in corners_cur]
        mask_cur = [H_view[r * bin_cnt_q + q] for r, q in corners_cur]

        Scur = bilinear_interpolation(
            bin_centers_q[curidx_q],  # left
            bin_centers_q[curidx_q_],  # right
            bin_centers_r[curidx_r],  # upper
            bin_centers_r[curidx_r_],  # lower
            value_cur,
            mask_cur,
            rcur,
            qcur,
        )

        corners_nxt = [
            (nxtidx_r, nxtidx_q),  # upper left
            (nxtidx_r, nxtidx_q_),  # upper right
            (nxtidx_r_, nxtidx_q),  # lower left
            (nxtidx_r_, nxtidx_q_),  # lower right
        ]

        value_nxt = [S_view[r * bin_cnt_q + q] for r, q in corners_nxt]
        mask_nxt = [H_view[r * bin_cnt_q + q] for r, q in corners_nxt]

        Snxt = bilinear_interpolation(
            bin_centers_q[nxtidx_q],  # left
            bin_centers_q[nxtidx_q_],  # right
            bin_centers_r[nxtidx_r],  # upper
            bin_centers_r[nxtidx_r_],  # lower
            value_nxt,
            mask_nxt,
            rnxt,
            qnxt,
        )

        gcur = np.log((rho_cur / pscur if posswt else (1 - rho_cur) / nscur))
        gnxt = np.log(((1 - rho_nxt) / nsnxt if posswt else rho_nxt / psnxt))

        # switch back
        if np.log(np.random.rand()) > Scur - Snxt + gnxt - gcur:
            SNet.switch(swt)
            ps, ns = pscur, nscur
            reject += 1
        else:
            reject = 0

        ps, ns = SNet.count_checker()
        idx_r = np.abs(
            bin_centers_r - SNet.assortativity_coeff()
        ).argmin()  #     np.searchsorted(bin_edges_r, SNet.assortativity_coeff(), side="right") - 1
        idx_q = np.abs(
            bin_centers_q - (ns - ps) / (ns + ps)
        ).argmin()  # np.searchsorted(bin_edges_q, (ns - ps) / (ns + ps), side="right") - 1
        # if idx_q <= 1:
        #    print((ns - ps) / (ns + ps))
        # iter_cnt -= 1
        q.put(
            (
                pid,
                idx_r * bin_cnt_q + idx_q,
                idx_r,
                idx_q,
            )
        )

    q.put(None)


if __name__ == "__main__":
    seed1, seed2 = 1, 1
    random.seed(seed1)
    np.random.seed(seed2)
    if sys.argv[1] == "NetRepo":
        # Experiments/network-repo/reptilia-tortoise-network-cs.edges
        G = nx.read_edgelist(
            "Experiments/network-repo/{:s}.edges".format(sys.argv[2]),
            nodetype=int,
            data=[("weight", float)],
        )
        A = nx.to_numpy_array(G, weight=None)
        A = (A != 0).astype(int)
        folder_path = "Experiments/Sampling/repo-{:s}".format(sys.argv[2])
        os.makedirs(folder_path, exist_ok=True)
        n = A.shape[0]
        des = graph_des(sys.argv[2], n, 0, seed1, seed2)
        # print(nx.degree_assortativity_coefficient(nx.karate_club_graph()))
    elif sys.argv[1] == "Karate":
        folder_path = "Experiments/Sampling/Karate"
        os.makedirs(folder_path, exist_ok=True)
        A = nx.adjacency_matrix(nx.karate_club_graph()).toarray()
        A = (A != 0).astype(int)
        n = A.shape[0]
        des = graph_des("Karate", n, 0, seed1, seed2)
        print(nx.degree_assortativity_coefficient(nx.karate_club_graph()))
    elif sys.argv[1] == "ER":
        n = int(sys.argv[2])  # 1024
        p = np.round(1.1 * np.log(n) / n, 2)
        net = ig.Graph.Erdos_Renyi(n=n, p=p)
        des = graph_des("ER", n, p, seed1, seed2)
        folder_path = "Experiments/Sampling/{:s}-{:d}-{:.2f}-{:d}-{:d}".format(
            des.type, des.n, des.k, des.npseed, des.seed
        )
        os.makedirs(folder_path, exist_ok=True)
        A = ig_to_A(net)
    elif sys.argv[1] == "Florentine":
        folder_path = "Experiments/Sampling/Florentine"
        os.makedirs(folder_path, exist_ok=True)
        A = nx.adjacency_matrix(nx.florentine_families_graph()).toarray()
        A = (A != 0).astype(int)
        n = A.shape[0]
        des = graph_des("Flo", n, 0, seed1, seed2)
        print(nx.degree_assortativity_coefficient(nx.florentine_families_graph()))
    else:
        raise "Not implenented"

    new_run = sys.argv[3] == "-1"
    if new_run:
        runid = 0
        while True:
            if os.path.isfile(os.path.join(folder_path, "{:d}.json".format(runid))):
                runid += 1
            else:
                break
    else:
        runid = int(sys.argv[3])
        with open(
            "{:s}/{:d}.json".format(folder_path, runid),
            "r",
        ) as file:
            data = json.load(file)

    SNet = MatSamp(A, False)
    print(SNet.assortativity_coeff())
    ps, ns = SNet.count_checker()
    # print(SNet.assortativity_coeff(), (ns - ps) / (ns + ps))
    r_0 = SNet.assortativity_coeff()
    q_0 = (ns - ps) / (ns + ps)
    step = 2 * (SNet.M3 - SNet.M1)
    # print(2*SNet.M1-SNet.M3, SNet.M3, step)
    # 0/0
    if new_run:
        bin_cnt_r = int(min(100, step))
        bin_cnt_q = bin_cnt_r
    else:
        bin_cnt_r = data["bin_cnt_r"]
        bin_cnt_q = data["bin_cnt_q"]
        samples_0 = data["sample_num"]

    print(bin_cnt_r, bin_cnt_q)
    bin_edges_r = np.linspace(-1, 1, bin_cnt_r + 1)
    bin_centers_r = (bin_edges_r[:-1] + bin_edges_r[1:]) / 2
    bin_delta_r = bin_centers_r[1] - bin_centers_r[0]

    bin_edges_q = np.linspace(-1, 1, bin_cnt_q + 1)
    bin_centers_q = (bin_edges_q[:-1] + bin_edges_q[1:]) / 2
    bin_delta_q = bin_centers_q[1] - bin_centers_q[0]

    sample_num = 1e9
    sampler_num = 60

    lock = mp.Lock()
    S = mp.Array("d", bin_cnt_r * bin_cnt_q)
    H = mp.Array("i", bin_cnt_r * bin_cnt_q)
    terminate_flag = mp.Value("i", 1)
    S_view = np.frombuffer(S.get_obj(), dtype=np.float64)
    H_view = np.frombuffer(H.get_obj(), dtype=np.int32)
    if new_run:
        S_view[:] = 1
    else:
        S_data = [item for row in data["S"] for item in row]
        H_data = [np.int32(item) for row in data["H"] for item in row]
        # print(len(H_data))
        S_view[:] = S_data
        H_view[:] = H_data
    # 0/0
    # print("Done loading")
    q = mp.Queue(maxsize=sampler_num)

    updater = mp.Process(
        target=updater,
        args=(
            q,
            S,
            H,
            sampler_num,
            bin_delta_r,
            bin_delta_q,
            terminate_flag,
            bin_cnt_r,
            bin_cnt_q,
        ),
    )
    updater.start()
    # print()
    samplers = []
    for i in range(sampler_num):
        SNet = MatSamp(A, False)
        SNet.trackCheckerboard = True
        # for _ in range(100*i):
        #     if i<sampler_num//2:
        #         swt = SNet.next("pos")
        #     else:
        #         swt = SNet.next("neg")
        #     SNet.switch(swt)
        samplers.append(
            mp.Process(
                target=sampler,
                args=(
                    q,
                    i,
                    SNet,
                    int(sample_num / sampler_num),
                    bin_centers_r,
                    bin_centers_q,
                    S,
                    H,
                    terminate_flag,
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
