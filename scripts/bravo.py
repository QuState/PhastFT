import itertools
import math
import time

import numpy as np


def reverse_bits(n, no_of_bits):
    result = 0
    for i in range(no_of_bits):
        result <<= 1
        result |= n & 1
        n >>= 1
    return result


def padded_bin(n, k):
    return bin(k)[2:].zfill(n)


def cobravo(x: list[int], y: list[int]) -> None:
    N = len(x)
    n = int(math.log2(N))
    W = 4
    L = 4
    B = 4
    assert N >= B**2
    assert B >= L >= W

    T = [0 for _ in range(B**2)]
    assert len(T) == B**2

    tau = int(N / B**2)
    assert tau >= 1

    rho = int(B / W)
    assert rho >= 1

    for t in range(0, tau):
        t_prime = reverse_bits(t, int(math.log2(n)))
        print(f"t: {t} t': {t_prime}")

        # copy sources to buffer
        for u in range(0, B):
            for v in range(0, B):
                # x_idx = (1 << u) | (1 << t) | (1 << v)  # u * t * v
                # t_idx = (1 << u) | (1 << v)  # u * v
                x_idx = u * t * v
                t_idx = u * v

                # print(f"index {x_idx} of x: {x[x_idx]}")
                T[t_idx] = x[x_idx]

        for l in range(0, rho**2):
            a = [T[i] for i in range(0, 4)]
            b = [T[i] for i in range(4, 8)]
            c = [T[i] for i in range(8, 12)]
            d = [T[i] for i in range(12, 16)]
            print(a, b, c, d)

            print("round 1")
            a, b = interleave(a, b)
            print(a, b)
            T[0:4] = a
            T[4:8] = b
            print(T)

            c, d = interleave(c, d)
            print(c, d)
            T[8:12] = c
            T[12:16] = d
            print(T)

            a = [T[i] for i in range(0, 4)]
            b = [T[i] for i in range(4, 8)]
            c = [T[i] for i in range(8, 12)]
            d = [T[i] for i in range(12, 16)]

            print("round 2")
            a, c = interleave(a, c)
            print(a, c)
            T[0:4] = a
            T[8:12] = c
            print(T)

            b, d = interleave(b, d)
            print(b, d)
            T[4:8] = c
            T[12:16] = d
            print(T)

            print(
                f"l: {l} T[B^2]: T[{B**2}]",
            )

        for u in range(0, B):
            for v in range(0, B):
                y_idx = u * t_prime * v
                t_idx = u * v
                y[y_idx] = T[t_idx]

    print(y)


def top_down_bril(x: list[int]) -> list[int]:
    if len(x) == 1:
        return x

    y = []
    evens = []
    odds = []

    for i in range(1, len(x), 2):
        evens.append(x[i - 1])
        odds.append(x[i])

    y += top_down_bril(evens)
    y += top_down_bril(odds)
    return y


def top_down_bril_iterative(x: list[int]) -> list[int]:
    if len(x) == 1:
        return x

    y, evens, odds = [], [], []

    for i in range(1, len(x), 2):
        evens.append(x[i - 1])
        odds.append(x[i])

    stack = [odds, evens]

    while stack:
        v = stack.pop()

        if len(v) == 1:
            y += v
            continue

        e, o = [], []

        for i in range(1, len(v), 2):
            e.append(v[i - 1])
            o.append(v[i])

        stack.append(o)
        stack.append(e)

    return y


def reverse_index_state(state: list[int]) -> list[int]:
    n = int(math.log2(len(state)))
    s = state.copy()
    for k in range(len(state)):
        s[k] = state[int(padded_bin(n, k)[::-1], 2)]
    return s


def apply_hadamard(state: list[complex], target: int) -> None:
    FRAC_1_SQRT2 = 1.0 / math.sqrt(2)
    h = [FRAC_1_SQRT2, FRAC_1_SQRT2, FRAC_1_SQRT2, -FRAC_1_SQRT2]
    n = len(state)

    num_pairs = n // 2
    dist = 1 << target

    for i in range(num_pairs):
        l0 = i + ((i >> target) << target)
        l1 = l0 + dist
        s0, s1 = state[l0], state[l1]
        state[l0] = s0 * h[0] + s1 * h[1]
        state[l1] = s0 * h[2] + s1 * h[3]


#
# for target in range(num_qubits):
#     apply_hadamard(state, target)


def apply_hadamard_target_0(state: list[complex]) -> None:
    FRAC_1_SQRT2 = 1.0 / math.sqrt(2)
    h = [FRAC_1_SQRT2, FRAC_1_SQRT2, FRAC_1_SQRT2, -FRAC_1_SQRT2]
    n = len(state)

    for i in range(1, n, 2):
        s0, s1 = state[i - 1], state[i]
        state[i - 1] = s0 * h[0] + s1 * h[1]
        state[i] = s0 * h[2] + s1 * h[3]


def apply_h_all_qubits(x: list[complex]) -> list[complex]:
    if len(x) == 1:
        return x

    apply_hadamard_target_0(x)

    y, evens, odds = [], [], []

    for i in range(1, len(x), 2):
        evens.append(x[i - 1])
        odds.append(x[i])

    apply_hadamard_target_0(evens)
    apply_hadamard_target_0(odds)

    stack = [odds, evens]

    while stack:
        v = stack.pop()

        if len(v) == 1:
            y += v
            continue

        e, o = [], []

        for i in range(1, len(v), 2):
            e.append(v[i - 1])
            o.append(v[i])

        apply_hadamard_target_0(e)
        apply_hadamard_target_0(o)

        stack.append(o)
        stack.append(e)

    return y


VECTOR_SIZE = 4


def interleave(a: list[int], b: list[int]) -> (list[int], list[int]):
    assert len(a) == VECTOR_SIZE and len(b) == VECTOR_SIZE
    c = [a[0], b[0], a[1], b[1]]
    d = [a[2], b[2], a[3], b[3]]
    return c, d


def bravo(x: list[int]) -> list[int]:
    # referred to as W in Anton's paper

    # load N/W vectors from x
    # N/W == len(x) / VECTOR_SIZE
    vecs = [
        [j for j in range(i, i + VECTOR_SIZE)] for i in range(0, len(x), VECTOR_SIZE)
    ]

    # number of classes is N/(W^2)
    # N / (W^2) == len(x) / VECTOR_SIZE**2 == len(x) / 16
    num_classes = len(x) // VECTOR_SIZE**2
    print(f"# of classes: {num_classes}")

    # group classes together
    classes = []
    for i in range(num_classes):
        classes.append([vecs[j] for j in range(i * VECTOR_SIZE, (i + 1) * VECTOR_SIZE)])

    for i in range(num_classes):
        print(f"class {i}: {classes[i]}")

    y = []

    print("now process each class, 'in parallel':")
    for i, c in enumerate(classes):
        v0, v1 = interleave(c[0], c[1])
        c[0], c[1] = v0, v1
        v2, v3 = interleave(c[2], c[3])
        c[2], c[3] = v2, v3

        v0, v1 = interleave(c[0], c[2])
        c[0], c[2] = v0, v1
        v2, v3 = interleave(c[1], c[3])
        c[1], c[3] = v2, v3

        print(f"class {i} is now: {c}")
        y.extend(itertools.chain.from_iterable(c))

    return y


def main() -> None:
    n = 5
    N = 1 << n
    x = [i for i in range(N)]
    y = bravo(x)
    print(y)


if __name__ == "__main__":
    # main()
    from pybindings import fft

    for n in range(4, 31):
        print(f"n = {n}")
        N = 1 << n
        a_re = [float(i) for i in range(N)]
        a_im = [float(i) for i in range(N)]

        start = time.time()
        a_re, a_im = fft(a_re, a_im)
        elapsed = (time.time() - start) * 10**6
        print(f"phastft python binding took: {elapsed} us")

        a = [complex(i, i) for i in range(N)]

        start = time.time()
        expected = np.fft.fft(a)
        elapsed = (time.time() - start) * 10**6
        print(f"numpy's fft took: {elapsed} us\n--------------------------------")

        actual = np.asarray(
            [
                complex(z_re, z_im)
                for (z_re, z_im) in zip(
                    a_re,
                    a_im,
                )
            ]
        )
        np.testing.assert_allclose(actual, expected)

        a = pyfftw.empty_aligned(128, dtype="complex128", n=16)

    # limit = 21
    # for n in range(2, limit):
    #     state0 = [complex(0.0, 0.0) for _ in range(2**n)]
    #     state0[0] = complex(1.0, 0.0)
    #
    #     start = time.time()
    #     state0 = apply_h_all_qubits(state0)
    #     elapsed = time.time() - start
    #     print(f"pairs pre-sorted applied H to n qubits in: {elapsed} s")
    #
    #     state1 = [complex(0.0, 0.0) for _ in range(2**n)]
    #     state1[0] = complex(1.0, 0.0)
    #
    #     start = time.time()
    #     for t in range(n):
    #         apply_hadamard(state1, t)
    #
    #     elapsed = time.time() - start
    #     print(f"insertion strategy applied H to {n} qubits in: {elapsed} s")
    #
    #     for a0, a1 in zip(state0, state1):
    #         assert math.isclose(a0.real, a1.real, rel_tol=1e-9, abs_tol=0.0)
    #         assert math.isclose(a0.imag, a1.imag, rel_tol=1e-9, abs_tol=0.0)
    #
    #     print(f"-----------------------------------------------------------")
