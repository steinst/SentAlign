"""The compiled functions of Vecalign (``_vecalign_core.pyx``) in NumPy and Python, used when the extension was not
built.  Same results up to floating-point rounding of the dot products (the compiled code adds the products of a
vector pair one by one in float32); the dynamic programmes are the same statements, only slower.

Written for SentAlign after ``vecalign/dp_core.pyx`` (Copyright 2019 Brian Thompson, Apache License 2.0).
"""
import numpy as np


def make_x_y_offsets(alignment_types):
    for x, y in alignment_types:
        assert x > 0 and y > 0
    return (np.array([x for x, y in alignment_types], dtype=np.int32),
            np.array([y for x, y in alignment_types], dtype=np.int32))


def make_dense_costs(vecs0, vecs1, norm0, norm1, offset0=0, offset1=0):
    sim = np.matmul(vecs0[offset0], vecs1[offset1].T).astype(np.float32)
    costs = np.float32(2.0) * (np.float32(1.0) - sim) / (np.float32(1e-6) + norm0[offset0][:, None] + norm1[offset1][None, :])
    return (costs * np.float32((offset0 + 1) * (offset1 + 1))).astype(np.float32)


def dense_dp(alignment_cost, pen):
    size0, size1 = alignment_cost.shape
    csum = np.empty((size0 + 1, size1 + 1), dtype=np.float64)
    bp = np.empty((size0 + 1, size1 + 1), dtype=np.int32)
    csum[0, :] = np.arange(size1 + 1) * pen
    bp[0, :] = 1
    csum[:, 0] = np.arange(size0 + 1) * pen
    bp[:, 0] = 2
    csum[0, 0] = 0.0
    bp[0, 0] = 4
    cost = alignment_cost.astype(np.float64)
    pen = float(pen)
    for c in range(1, size1 + 1):
        prev_col, col = csum[:, c - 1], csum[:, c]
        for r in range(1, size0 + 1):
            best, b = prev_col[r - 1] + cost[r - 1, c - 1], 0
            cost1 = prev_col[r] + pen
            if cost1 < best:
                best, b = cost1, 1
            cost2 = col[r - 1] + pen
            if cost2 < best:
                best, b = cost2, 2
            col[r] = best
            bp[r, c] = b
    return csum, bp


def score_path(xx, yy, norm1, norm2, vecs1, vecs2, out):
    dots = np.einsum('ij,ij->i', vecs1[xx], vecs2[yy]).astype(np.float32)
    out[:] = np.float32(2.0) * (np.float32(1.0) - dots) / (norm1[xx] + norm2[yy])


def make_sparse_costs(vecs0, vecs1, norms0, norms1, x_y_path, alignment_types, width_over2):
    path = np.array(x_y_path).astype(np.int32)
    max_x = max([0] + [x for x, y in alignment_types])
    max_y = max([0] + [y for x, y in alignment_types])
    if max_x > vecs0.shape[0]:
        raise Exception('%d x overlaps requested (via alignment_types), but vecs0 only has %d' % (max_x, vecs0.shape[0]))
    if max_y > vecs1.shape[0]:
        raise Exception('%d y overlaps requested (via alignment_types), but vecs1 only has %d' % (max_y, vecs1.shape[0]))
    xsize, ysize = vecs0.shape[1], vecs1.shape[1]
    a_len, b_len = path.shape[0], 2 * width_over2
    feats = np.full((len(alignment_types), a_len, b_len), np.inf, dtype=np.float32)
    aa = path[:, 0] + path[:, 1]
    b_offset = np.empty(a_len, dtype=np.int32)
    b_offset[aa] = path[:, 1] - width_over2
    yy = (path[:, 1] - width_over2)[:, None] + np.arange(b_len)[None, :]         # (a_len, b_len)
    xx = aa[:, None] - yy
    ok = (xx >= 0) & (xx < xsize) & (yy >= 0) & (yy < ysize)
    for start in range(0, a_len, 1024):                                           # in blocks: memory
        sl = slice(start, start + 1024)
        rows, cols = np.nonzero(ok[sl])
        xi, yi = xx[sl][rows, cols], yy[sl][rows, cols]
        for t, (xo, yo) in enumerate(alignment_types):
            dots = np.einsum('ij,ij->i', vecs0[xo - 1, xi], vecs1[yo - 1, yi]).astype(np.float32)
            val = np.float32(2.0 * xo * yo) * (np.float32(1.0) - dots) / (
                np.float32(1e-6) + norms0[xo - 1, xi] + norms1[yo - 1, yi])
            feats[t, aa[sl][rows], cols] = val
    return feats, b_offset


def sparse_dp(a_b_costs, b_offset_in, alignment_types, del_penalty, x_in_size, y_in_size):
    x_offsets, y_offsets = make_x_y_offsets(alignment_types)
    moves = list(zip(x_offsets.tolist() + [0, 1], y_offsets.tolist() + [1, 0]))
    a_in_size, b_in_size = a_b_costs.shape[1], a_b_costs.shape[2]
    a_out_size, b_out_size = a_in_size + 2, b_in_size
    x_out_size, y_out_size = x_in_size + 1, y_in_size + 1
    b_offset_out = np.concatenate([np.array([b_offset_in[0], b_offset_in[0]], dtype=np.int32), b_offset_in + 1])
    csum = np.full((a_out_size, b_in_size), np.inf, dtype=np.float64)
    xp = np.full((a_out_size, b_in_size), -2, dtype=np.int32)
    yp = np.full((a_out_size, b_in_size), -2, dtype=np.int32)
    off_in, off_out = b_offset_in.tolist(), b_offset_out.tolist()
    costs = a_b_costs.astype(np.float64)
    del_penalty = float(del_penalty)
    inf = float('inf')
    for aa_out in range(a_out_size):
        for bb_out in range(b_in_size):
            yy_out = bb_out + off_out[aa_out]
            xx_out = aa_out - yy_out
            if xx_out == 0 and 0 <= yy_out < y_out_size:
                csum[aa_out, bb_out] = del_penalty * yy_out
                xp[aa_out, bb_out], yp[aa_out, bb_out] = 0, 1
            elif yy_out == 0 and 0 <= xx_out < x_out_size:
                csum[aa_out, bb_out] = del_penalty * xx_out
                xp[aa_out, bb_out], yp[aa_out, bb_out] = 1, 0
            else:
                best, bx, by = inf, -42, -42
                xx_in, yy_in = xx_out - 1, yy_out - 1
                if 0 <= xx_in < x_in_size and 0 <= yy_in < y_in_size:
                    aa_in = xx_in + yy_in
                    bb_in = yy_in - off_in[aa_in] if 0 <= aa_in < a_in_size else -1
                    if 0 <= aa_in < a_in_size and 0 <= bb_in < b_in_size:
                        for ii, (xo, yo) in enumerate(moves):
                            xx_prev, yy_prev = xx_out - xo, yy_out - yo
                            if not (0 <= xx_prev < x_out_size and 0 <= yy_prev < y_out_size):
                                continue
                            aa_prev = xx_prev + yy_prev
                            if not 0 <= aa_prev < a_out_size:
                                continue
                            bb_prev = yy_prev - off_out[aa_prev]
                            if not 0 <= bb_prev < b_out_size:
                                continue
                            step = del_penalty if (xo == 0 or yo == 0) else costs[ii, aa_in, bb_in]
                            total = csum[aa_prev, bb_prev] + step
                            if total < best:
                                best, bx, by = total, xo, yo
                csum[aa_out, bb_out] = best
                xp[aa_out, bb_out], yp[aa_out, bb_out] = bx, by
    return csum, xp, yp, b_offset_out
