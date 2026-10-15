"""Minimum-snap polynomial trajectories (Mellinger & Kumar, ICRA 2011).

Each segment is a 7th-order polynomial per axis. We minimise the integral of squared snap subject to
waypoint positions, zero velocity/acceleration/jerk at both ends, and continuity of derivatives 1-6
at interior waypoints. The equality-constrained QP is solved exactly through its KKT system.
Segment times can be optimised with the time-penalised cost J(T) + k_T * sum(T).
"""
import math
import numpy as np

N = 8  # coefficients per segment (7th order)


def _deriv_row(t, k):
    """Row vector r so that r @ c = k-th derivative at time t of sum c_i t^i."""
    r = np.zeros(N)
    for i in range(k, N):
        r[i] = math.factorial(i) / math.factorial(i - k) * t ** (i - k)
    return r


def _snap_cost_matrix(T):
    """Q such that c^T Q c = integral_0^T (4th derivative)^2 dt."""
    Q = np.zeros((N, N))
    for i in range(4, N):
        for j in range(4, N):
            ci = math.factorial(i) / math.factorial(i - 4)
            cj = math.factorial(j) / math.factorial(j - 4)
            Q[i, j] = ci * cj * T ** (i + j - 7) / (i + j - 7)
    return Q


def solve_axis(points, times):
    """points: (m+1,) waypoint values for one axis. times: (m,) segment durations."""
    m = len(times)
    nv = N * m
    H = np.zeros((nv, nv))
    for s, T in enumerate(times):
        H[s * N:(s + 1) * N, s * N:(s + 1) * N] = _snap_cost_matrix(T)
    A, b = [], []

    def add(row_blocks, val):
        row = np.zeros(nv)
        for seg, vec in row_blocks:
            row[seg * N:(seg + 1) * N] += vec
        A.append(row); b.append(val)

    for s, T in enumerate(times):
        add([(s, _deriv_row(0, 0))], points[s])          # start of segment at waypoint s
        add([(s, _deriv_row(T, 0))], points[s + 1])      # end of segment at waypoint s+1
    for k in (1, 2, 3):                                   # rest at both ends
        add([(0, _deriv_row(0, k))], 0.0)
        add([(m - 1, _deriv_row(times[-1], k))], 0.0)
    for s in range(m - 1):                                # smooth joins
        for k in range(1, 7):
            add([(s, _deriv_row(times[s], k)), (s + 1, -_deriv_row(0, k))], 0.0)
    A, b = np.array(A), np.array(b)
    kkt = np.block([[2 * H, A.T], [A, np.zeros((len(b), len(b)))]])
    rhs = np.concatenate([np.zeros(nv), b])
    sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    c = sol[:nv].reshape(m, N)
    cost = float(sol[:nv] @ H @ sol[:nv])
    return c, cost


class Trajectory:
    def __init__(self, waypoints, times):
        self.wp = np.asarray(waypoints, float)
        self.times = np.asarray(times, float)
        self.t_knots = np.concatenate([[0.0], np.cumsum(self.times)])
        self.coef, self.cost = [], 0.0
        for ax in range(3):
            c, cost = solve_axis(self.wp[:, ax], self.times)
            self.coef.append(c)
            self.cost += cost

    @property
    def duration(self):
        return float(self.t_knots[-1])

    def eval(self, t, k=0):
        """k-th derivative of position at time t, shape (3,)."""
        t = min(max(t, 0.0), self.duration)
        s = min(int(np.searchsorted(self.t_knots, t, side="right") - 1), len(self.times) - 1)
        tau = t - self.t_knots[s]
        r = _deriv_row(tau, k)
        return np.array([self.coef[ax][s] @ r for ax in range(3)])


def allocate_times(waypoints, avg_speed):
    d = np.linalg.norm(np.diff(np.asarray(waypoints, float), axis=0), axis=1)
    return d / avg_speed


def optimize_times(waypoints, times0, k_time, iters=60):
    """Minimise snap cost + k_time * total time by projected gradient descent on log segment times."""
    logT = np.log(np.asarray(times0, float))

    def J(lt):
        T = np.exp(lt)
        return Trajectory(waypoints, T).cost + k_time * T.sum()

    best = J(logT)
    step = 0.1
    for _ in range(iters):
        g = np.zeros_like(logT)
        for i in range(len(logT)):
            e = np.zeros_like(logT); e[i] = 1e-4
            g[i] = (J(logT + e) - J(logT - e)) / 2e-4
        cand = logT - step * g / (np.linalg.norm(g) + 1e-12)
        jc = J(cand)
        if jc < best:
            logT, best, step = cand, jc, step * 1.2
        else:
            step *= 0.5
            if step < 1e-4:
                break
    return np.exp(logT)
