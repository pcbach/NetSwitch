from MatSamp import *
import matplotlib.pyplot as plt
import igraph as ig
import pickle
import json
import multiprocessing as mp
import time
from numpy.random import default_rng
from collections import namedtuple
import os

# from multiprocessing import Process, Queue
import random

# print(mp.cpu_count())
plt.rcParams.update(
    {"text.usetex": True, "font.family": "STIXGeneral", "mathtext.fontset": "stix"}
)


graph_des = namedtuple("graph_des", ["type", "n", "k", "npseed", "seed"])
des = graph_des("None", "None", "None", "None", "None")


def bilinear_interpolation(
    left_col,
    right_col,
    upper_row,
    lower_row,
    upper_left_val,
    upper_right_val,
    lower_left_val,
    lower_right_val,
    row,
    col,
):
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


# print(bilinear_interpolation(0,2,2,1,4,5,2,4,1.5,1))
# 0/0
def gaussian_kernel(size=3, sigma=1.0):
    ax = np.arange(-(size // 2), size // 2 + 1)
    X, Y = np.meshgrid(ax, ax)
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


k_size = 3
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
    H_star = np.zeros(bin_cnt_q*bin_cnt_r)
    f = 2 ** (4)
    samples = 0
    finished = 0
    tunnelflag = np.zeros(sampler_num)
    min_q = 1e9
    max_q = -1e9
    tunnel = 0
    latest_tunnel = 0

    while finished < sampler_num:
        data = q.get()

        if data is None:
            finished += 1
        else:
            pid, idx, idx_r, idx_q = data
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

            H_view[idx] += 1
            H_star[idx] += 1
            samples += 1

            for dr in range(-(k_size//2),(k_size//2)):
                for dq in range(-(k_size//2),(k_size//2)):
                    idx_r_, idx_q_ = idx_r +dr , idx_q+dq
                    idx_ = idx_r_*bin_cnt_q+idx_q_
                    if 0 <= idx_r_ < bin_cnt_r and 0 <= idx_q_ < bin_cnt_q and H_star[idx_] != 0:
                        S_view[idx_] += f * kernel[(k_size//2)+dr,(k_size//2)+dq] 
            # S_view[idx] += f  # / delta_q / delta_r

            if tunnel >= 1:

                while finished < sampler_num and not q.empty():
                    data = q.get()
                    if data is None:
                        finished += 1
                    else:
                        pid, idx, idx_r, idx_q = data
                        H_view[idx] += 1
                        H_star[idx] += 1
                        samples += 1
                        S_view[idx] += f
                        for dr in range(-(k_size//2),(k_size//2)):
                            for dq in range(-(k_size//2),(k_size//2)):
                                idx_r_, idx_q_ = idx_r +dr , idx_q+dq
                                idx_ = idx_r_*bin_cnt_q+idx_q_
                                if 0 <= idx_r_ < bin_cnt_r and 0 <= idx_q_ < bin_cnt_q and H_star[idx_] != 0:
                                    S_view[idx_] += f * kernel[(k_size//2)+dr,(k_size//2)+dq] 

                tunnelflag = np.zeros(sampler_num)
                tunnel = 0
                latest_tunnel = 0
                H_view[:] = 0
                min_q = 1e9
                max_q = -1e9

                f = f / 2

                folder_path = "Experiments/{:s}-{:d}-{:.2f}-{:d}-{:d}".format(
                    des.type, des.n, des.k, des.npseed, des.seed
                )
                os.makedirs(folder_path, exist_ok=True)
                print("{:d}: Step size decreased, f={:.3e}".format(samples, f))
                filename = "{:s}-{:d}-{:.2f}-{:d}-{:d}/{:d}".format(
                    des.type, des.n, des.k, des.npseed, des.seed, samples
                )
                with open("Experiments/" + filename + ".json", "w") as ffff:
                    json.dump(
                        {
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
                if f < 2 ** (-8):
                    terminate_flag.value = 0

        if samples % 1e3 == 0:
            filename = "{:s}-{:d}-{:.2f}-{:d}-{:d}".format(
                des.type,
                des.n,
                des.k,
                des.npseed,
                des.seed,
            )
            print(samples, ": ", end="")
            print(
                "{:.3e} {:.1f} {:.0f} {:.0f}".format(f, tunnel, min_q, max_q),
            )
            # print(tunnelflag)
            # # with open("Experiments/" + filename + ".pkl", "wb") as ffff:
            # #   pickle.dump((des, S_view, H_view), ffff)=
        if samples % 1e6 == 0 or samples == sampler_num:
            folder_path = "Experiments/{:s}-{:d}-{:.2f}-{:d}-{:d}".format(
                des.type, des.n, des.k, des.npseed, des.seed
            )
            os.makedirs(folder_path, exist_ok=True)
            filename = "{:s}-{:d}-{:.2f}-{:d}-{:d}/{:d}".format(
                des.type, des.n, des.k, des.npseed, des.seed, samples
            )
            with open("Experiments/" + filename + ".json", "w") as ffff:
                json.dump(
                    {
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
    bin_edges_r,
    bin_centers_r,
    bin_edges_q,
    bin_centers_q,
    S,
    terminate_flag,
):
    bin_cnt_r = len(bin_centers_r)
    bin_cnt_q = len(bin_centers_q)
    rng = default_rng()

    while iter_cnt and terminate_flag.value:

        pscur, nscur = SNet.count_checker()
        ps, ns = pscur, nscur
        qcur = (nscur - pscur) / (nscur + pscur)
        rcur = SNet.assortativity_coeff()

        kappa = 3
        # #rho_cur = (1 - qcur) / 2
        # #rho_cur = ((1 - qcur) / 2)
        # print(qcur, rho_cur)
        # p_posswt = rng.random()
        # posswt = p_posswt < rho_cur
        if pscur == 0:
            rho_cur = 0
            posswt = False
        elif nscur == 0:
            rho_cur = 1
            posswt = True
        else:
            #rho_cur = 0.5
            rho_cur = (kappa / 2 - qcur) / kappa
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

        if psnxt == 0:
            rho_nxt = 0
        elif nsnxt == 0:
            rho_nxt = 1
        else:
            #rho_nxt = 0.5
            rho_nxt = (kappa / 2 - qnxt) / kappa

        curidx_r = np.searchsorted(bin_centers_r, rcur, side="right") - 1
        nxtidx_r = np.searchsorted(bin_centers_r, rnxt, side="right") - 1

        curidx_q = np.searchsorted(bin_centers_q, qcur, side="right") - 1
        nxtidx_q = np.searchsorted(bin_centers_q, qnxt, side="right") - 1

        S_view = np.frombuffer(S.get_obj(), dtype=np.float64)

        assert any(S_view) != np.inf

        curidx_q_ = min(curidx_q + 1, bin_cnt_q - 1)
        curidx_r_ = min(curidx_r + 1, bin_cnt_r - 1)
        nxtidx_q_ = min(nxtidx_q + 1, bin_cnt_q - 1)
        nxtidx_r_ = min(nxtidx_r + 1, bin_cnt_r - 1)

        Scur = bilinear_interpolation(
            bin_centers_q[curidx_q],  # left
            bin_centers_q[curidx_q_],  # right
            bin_centers_r[curidx_r],  # upper
            bin_centers_r[curidx_r_],  # lower
            S_view[curidx_r * bin_cnt_q + curidx_q],  # upper left
            S_view[curidx_r * bin_cnt_q + curidx_q_],  # upper right
            S_view[curidx_r_ * bin_cnt_q + curidx_q],  # lower left
            S_view[curidx_r_ * bin_cnt_q + curidx_q_],  # lower right,
            rcur,
            qcur,
        )

        Snxt = bilinear_interpolation(
            bin_centers_q[nxtidx_q],  # left
            bin_centers_q[nxtidx_q_],  # right
            bin_centers_r[nxtidx_r],  # upper
            bin_centers_r[nxtidx_r_],  # lower
            S_view[nxtidx_r * bin_cnt_q + nxtidx_q],  # upper left
            S_view[nxtidx_r * bin_cnt_q + nxtidx_q_],  # upper right
            S_view[nxtidx_r_ * bin_cnt_q + nxtidx_q],  # lower left
            S_view[nxtidx_r_ * bin_cnt_q + nxtidx_q_],  # lower right,
            rnxt,
            qnxt,
        )

        gcur = np.log((rho_cur / pscur if posswt else (1 - rho_cur) / nscur))
        gnxt = np.log(((1 - rho_nxt) / nsnxt if posswt else rho_nxt / psnxt))

        # switch back
        if np.log(np.random.rand()) > Scur - Snxt + gnxt - gcur:
            SNet.switch(swt)
            ps, ns = pscur, nscur

        idx_r = (
            np.searchsorted(bin_edges_r, SNet.assortativity_coeff(), side="right") - 1
        )
        idx_q = np.searchsorted(bin_edges_q, (ns - ps) / (ns + ps), side="right") - 1

        iter_cnt -= 1

        q.put(
            (
                pid,
                idx_r * bin_cnt_q + idx_q,idx_r,
                idx_q,
            )
        )

    q.put(None)


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    n = 128
    p = np.round(1.1 * np.log(n) / n, 2)
    net = ig.Graph.Erdos_Renyi(n=n, p=p)
    des = graph_des("ER", n, p, 1, 1)

    A = ig_to_A(net)
    SNet = MatSamp(A, False)
    step = 2 * (SNet.M3 - SNet.M1)

    bin_cnt_r = int(min(100, step))
    bin_cnt_q = 100
    print(bin_cnt_r)
    bin_edges_r = np.linspace(-1, 1, bin_cnt_r + 1)
    bin_centers_r = (bin_edges_r[:-1] + bin_edges_r[1:]) / 2
    bin_delta_r = bin_centers_r[1] - bin_centers_r[0]

    bin_edges_q = np.linspace(-1, 1, bin_cnt_q + 1)
    bin_centers_q = (bin_edges_q[:-1] + bin_edges_q[1:]) / 2
    bin_delta_q = bin_centers_q[1] - bin_centers_q[0]

    sample_num = 1e8
    sampler_num = 50

    lock = mp.Lock()
    S = mp.Array("d", bin_cnt_r * bin_cnt_q)
    H = mp.Array("i", bin_cnt_r * bin_cnt_q)
    terminate_flag = mp.Value("i", 1)
    S_view = np.frombuffer(S.get_obj(), dtype=np.float64)
    S_view[:] = 0

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
                    bin_edges_r,
                    bin_centers_r,
                    bin_edges_q,
                    bin_centers_q,
                    S,
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
