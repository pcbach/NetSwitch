import numpy as np
import matplotlib.pyplot as plt
import json
import sys

plt.rcParams.update(
    {"text.usetex": True, "font.family": "STIXGeneral", "mathtext.fontset": "stix"}
)


def logsumlog(A):
    A_max = np.ma.max(A)
    exp_shifted = np.exp(A - A_max)
    s = np.sum(exp_shifted)
    if s <= 0 or not np.isfinite(s):
        return A_max
    return A_max + np.log(s)


with open("Experiments/" + sys.argv[1] + ".json", "r") as f:
    data = json.load(f)

bin_cnt_r = data["bin_cnt_r"]
bin_cnt_q = data["bin_cnt_q"]
S = np.array(data["S"])
H = np.array(data["H"])

bin_edges_r = np.linspace(-1, 1, bin_cnt_r + 1)
bin_centers_r = (bin_edges_r[:-1] + bin_edges_r[1:]) / 2
bin_delta_r = bin_centers_r[1] - bin_centers_r[0]

bin_edges_q = np.linspace(-1, 1, bin_cnt_q + 1)
bin_centers_q = (bin_edges_q[:-1] + bin_edges_q[1:]) / 2
bin_delta_q = bin_centers_q[1] - bin_centers_q[0]

# S = np.frombuffer(S.get_obj(), dtype=np.float64)
# H = np.frombuffer(H.get_obj(), dtype=np.int32)

outer_pad = 2
mid_pad = 4
pad = 1
main = 9
minor = 3
cbar_width = 0.5
cbar_pad = 0.5
width = (
    2 * main
    + 2 * minor
    + 2 * pad
    + mid_pad
    + 2 * outer_pad
    + 2 * cbar_width
    + 2 * cbar_pad
)
height = main + minor + pad + 2 * outer_pad
fig = plt.figure(figsize=(12, 12 * height / width))

S_2d = S.reshape(bin_cnt_r, bin_cnt_q)
H_2d = H.reshape(bin_cnt_r, bin_cnt_q)

cmap = plt.cm.magma.copy()
cmap.set_bad(color="white")

H_masked = np.ma.masked_where(H_2d == 0, H_2d)
S_masked = np.ma.masked_where(S_2d == 0, S_2d)
#print(np.mean(H_masked), np.max(H_masked), np.min(H_masked), np.std(H_masked))
#print(np.min(S_masked))
#S_masked = S_masked-np.min(S_masked)+1e-9
main_ax_S = fig.add_axes(
    [
        (outer_pad + minor + pad) / width,
        (outer_pad + minor + pad) / height,
        main / width,
        main / height,
    ]
)

im1 = main_ax_S.imshow(
    S_masked.T,
    extent=[-1, 1, -1, 1],
    cmap=cmap,
)
im1.set_clim(np.min(S_masked), np.max(S_masked))
# main_ax_S.set_xlabel(r"$r(g)$")
# main_ax_S.set_ylabel(r"$q(g)$")
# main_ax_S.set_xticks(np.linspace(-1, 1, 5))
# main_ax_S.set_yticks(np.linspace(-1, 1, 5))
main_ax_S.set_xticklabels([])
main_ax_S.set_yticklabels([])
# main_ax_S.spines["top"].set_visible(False)
# main_ax_S.spines["right"].set_visible(False)
# main_ax_S.spines["bottom"].set_visible(False)
# main_ax_S.spines["left"].set_visible(False)

Scbar = fig.add_axes(
    [
        (outer_pad + minor + pad + main + cbar_pad) / width,
        (outer_pad + minor + pad) / height,
        cbar_width / width,
        main / height,
    ]
)
cbS = fig.colorbar(im1, cax=Scbar)

main_ax_H = fig.add_axes(
    [
        (outer_pad + 2 * minor + 2 * pad + mid_pad + main + cbar_width + cbar_pad)
        / width,
        (outer_pad + minor + pad) / height,
        main / width,
        main / height,
    ]
)
im2 = main_ax_H.imshow(
    H_masked.T,
    extent=[-1, 1, -1, 1],
    cmap=cmap,
)
im2.set_clim(np.min(H_masked), np.max(H_masked))
# main_ax_H.set_xlabel(r"$r(g)$")
# main_ax_H.set_ylabel(r"$q(g)$")
# main_ax_H.set_xticks(np.linspace(-1, 1, 5))
# main_ax_H.set_yticks(np.linspace(-1, 1, 5))
main_ax_H.set_xticklabels([])
main_ax_H.set_yticklabels([])
# main_ax_H.spines["top"].set_visible(False)
# main_ax_H.spines["right"].set_visible(False)
# main_ax_H.spines["bottom"].set_visible(False)
# main_ax_H.spines["left"].set_visible(False)

Hcbar = fig.add_axes(
    [
        (
            outer_pad
            + 2 * minor
            + 2 * pad
            + mid_pad
            + 2 * main
            + cbar_width
            + 2 * cbar_pad
        )
        / width,
        (outer_pad + minor + pad) / height,
        cbar_width / width,
        main / height,
    ]
)
cbH = fig.colorbar(im2, cax=Hcbar)

# S_sum_row = S_masked.sum(axis=1)  # r
# S_sum_col = S_masked.sum(axis=0)  # q
H_sum_row = H_masked.sum(axis=1)  # r
H_sum_col = H_masked.sum(axis=0)  # q
S_sum_row = np.array([logsumlog(S_masked[i, :]) for i in range(bin_cnt_r)])
S_sum_col = np.array([logsumlog(S_masked[:, j]) for j in range(bin_cnt_q)])

minor_ax_S_y = fig.add_axes(
    [
        (outer_pad) / width,
        (outer_pad + minor + pad) / height,
        minor / width,
        main / height,
    ]
)
minor_ax_S_y.plot(S_sum_col, bin_centers_q[H_sum_col != 0])
minor_ax_S_y.set_yticks(np.linspace(-1, 1, 5))
minor_ax_S_y.set_ylim([-1, 1])
minor_ax_S_y.set_xlabel(r"$\log P_q(q) $")
minor_ax_S_y.set_ylabel(r"$q$")
minor_ax_S_y.spines["top"].set_visible(False)
minor_ax_S_y.spines["right"].set_visible(False)
minor_ax_S_y.invert_yaxis()

minor_ax_S_x = fig.add_axes(
    [
        (outer_pad + minor + pad) / width,
        (outer_pad) / height,
        main / width,
        minor / height,
    ]
)
minor_ax_S_x.plot(bin_centers_r[H_sum_row != 0], S_sum_row)
minor_ax_S_x.set_xlim([-1, 1])
minor_ax_S_x.set_xticks(np.linspace(-1, 1, 5))
minor_ax_S_x.set_xlabel(r"$r$")
minor_ax_S_x.set_ylabel(r"$\log P_r(r) $")
minor_ax_S_x.spines["top"].set_visible(False)
minor_ax_S_x.spines["right"].set_visible(False)

minor_ax_H_y = fig.add_axes(
    [
        (cbar_width + outer_pad + minor + pad + mid_pad + main + cbar_pad) / width,
        (outer_pad + minor + pad) / height,
        minor / width,
        main / height,
    ]
)
minor_ax_H_y.plot(H_sum_col, bin_centers_q[H_sum_col != 0])
# minor_ax_H_y.invert_yaxis()
minor_ax_H_y.set_yticks(np.linspace(-1, 1, 5))
minor_ax_H_y.set_ylim([-1, 1])
minor_ax_H_y.set_xlabel(r"\# samples")
minor_ax_H_y.set_ylabel(r"$q$")
minor_ax_H_y.spines["top"].set_visible(False)
minor_ax_H_y.spines["right"].set_visible(False)
minor_ax_H_y.invert_yaxis()

minor_ax_H_x = fig.add_axes(
    [
        (outer_pad + 2 * minor + 2 * pad + mid_pad + main + cbar_pad + cbar_width)
        / width,
        (outer_pad) / height,
        main / width,
        minor / height,
    ]
)
minor_ax_H_x.plot(bin_centers_r[H_sum_row != 0], H_sum_row)
minor_ax_H_x.set_xticks(np.linspace(-1, 1, 5))
minor_ax_H_x.set_xlim([-1, 1])
minor_ax_H_x.set_xlabel(r"r")
minor_ax_H_x.set_ylabel(r"\# samples")
minor_ax_H_x.spines["top"].set_visible(False)
minor_ax_H_x.spines["right"].set_visible(False)

# ax[1][0].set_axis_off()
# ax[1][2].set_axis_off()

# fig.align_ylabels([ax[0][1], ax[0][0], ax[0][2], ax[0][3]])
# fig.align_xlabels([ax[0][1], ax[1][1]])
# fig.align_xlabels([ax[0][2], ax[0][3]])
# plt.tight_layout()
plt.savefig("Experiments/mulproc_2d_1.pdf", dpi=1000)
