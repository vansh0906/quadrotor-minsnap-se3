"""Quadrotor rigid-body simulator and controllers.

- Dynamics on SE(3): position, velocity, rotation matrix R, body rates; 4 motors with thrust limits
  and a first-order motor lag; optional wind force.
- GeometricController: Lee, Leok & McClamroch (CDC 2010) SE(3) tracking control, with
  differential-flatness feed-forward of body rates and angular acceleration.
- EulerPDController: the common small-angle cascaded controller (hover linearisation), as baseline.
"""
import numpy as np

G = 9.81
E3 = np.array([0.0, 0.0, 1.0])


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def expm_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3) + hat(w)
    K = hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def rot_x(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


class Params:
    m = 1.0                                  # kg
    J = np.diag([0.0082, 0.0082, 0.0149])    # kg m^2
    arm = 0.17                               # m
    c_tau = 0.016                            # yaw torque per newton of thrust
    f_max = 6.0                              # N per motor (thrust-to-weight about 2.4)
    tau_motor = 0.02                         # s


class Quadrotor:
    def __init__(self, p=Params(), x0=None, R0=None):
        self.p = p
        self.x = np.zeros(3) if x0 is None else np.array(x0, float)
        self.v = np.zeros(3)
        self.R = np.eye(3) if R0 is None else np.array(R0, float)
        self.w = np.zeros(3)
        self.f = np.full(4, p.m * G / 4)
        a = p.arm / np.sqrt(2)
        xs, ys = np.array([a, -a, -a, a]), np.array([a, a, -a, -a])
        spin = np.array([1, -1, 1, -1])
        self.A = np.vstack([np.ones(4), ys, -xs, spin * p.c_tau])   # [f, Mx, My, Mz] = A @ motor thrusts
        self.A_inv = np.linalg.inv(self.A)
        self.saturated = 0

    def step(self, f_total, M, dt, wind=np.zeros(3)):
        cmd = self.A_inv @ np.concatenate([[f_total], M])
        clipped = np.clip(cmd, 0.0, self.p.f_max)
        self.saturated += int(np.any(np.abs(clipped - cmd) > 1e-9))
        self.f += (clipped - self.f) * min(1.0, dt / self.p.tau_motor)
        u = self.A @ self.f
        acc = (u[0] * self.R @ E3 + wind) / self.p.m - G * E3
        J = self.p.J
        wdot = np.linalg.solve(J, u[1:] - np.cross(self.w, J @ self.w))
        self.x = self.x + self.v * dt + 0.5 * acc * dt * dt
        self.v = self.v + acc * dt
        self.R = self.R @ expm_so3(self.w * dt + 0.5 * wdot * dt * dt)
        self.w = self.w + wdot * dt

    def tilt_deg(self):
        return float(np.degrees(np.arccos(np.clip(self.R[2, 2], -1, 1))))


def flat_reference(traj, t, yaw=0.0, dt=1e-3):
    """Desired position, velocity, acceleration, rotation, body rates and angular acceleration
    from the flat outputs (x, y, z, yaw) via differential flatness."""
    def att(tt):
        a = traj.eval(tt, 2)
        j = traj.eval(tt, 3)
        zb = a + G * E3
        c = np.linalg.norm(zb)
        zb = zb / c
        xc = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        yb = np.cross(zb, xc); yb /= np.linalg.norm(yb)
        xb = np.cross(yb, zb)
        h = (j - (zb @ j) * zb) / c
        w = np.array([-h @ yb, h @ xb, 0.0])
        return np.column_stack([xb, yb, zb]), w
    Rd, wd = att(t)
    _, wd2 = att(t + dt)
    return dict(x=traj.eval(t), v=traj.eval(t, 1), a=traj.eval(t, 2), R=Rd, w=wd, wdot=(wd2 - wd) / dt, yaw=yaw)


class GeometricController:
    name = "Geometric SE(3) + feed-forward"

    def __init__(self, p=Params(), feedforward=True, large_angle=True, kx=6.0, kv=4.5, kR=225.0, kW=27.0):
        self.p, self.ff, self.large = p, feedforward, large_angle
        self.kx, self.kv, self.kR, self.kW = kx, kv, kR, kW
        if not feedforward:
            self.name = "Geometric SE(3), feedback only"
        elif not large_angle:
            self.name = "Geometric SE(3), textbook e_R"

    def __call__(self, q, ref):
        m, J = self.p.m, self.p.J
        ex, ev = q.x - ref["x"], q.v - ref["v"]
        a_ff = ref["a"] if self.ff else np.zeros(3)
        F = m * (-self.kx * ex - self.kv * ev + G * E3 + a_ff)
        b3 = F / np.linalg.norm(F)
        b1d = np.array([np.cos(ref["yaw"]), np.sin(ref["yaw"]), 0.0])
        b2 = np.cross(b3, b1d); b2 /= np.linalg.norm(b2)
        Rc = np.column_stack([np.cross(b2, b3), b2, b3])
        f = F @ q.R @ E3
        if self.large:
            # error function from Lee's follow-up work: |e_R| = 2 sin(theta/2), still strong near 180 deg
            eR = vee(Rc.T @ q.R - q.R.T @ Rc) / np.sqrt(max(1.0 + np.trace(Rc.T @ q.R), 1e-3))
        else:
            # original CDC 2010 error: |e_R| = sin(theta), which fades to zero as theta -> 180 deg
            eR = 0.5 * vee(Rc.T @ q.R - q.R.T @ Rc)
        Wc = ref["w"] if self.ff else np.zeros(3)
        Wdc = ref["wdot"] if self.ff else np.zeros(3)
        eW = q.w - q.R.T @ Rc @ Wc
        M = J @ (-self.kR * eR - self.kW * eW) + np.cross(q.w, J @ q.w) \
            - J @ (hat(q.w) @ q.R.T @ Rc @ Wc - q.R.T @ Rc @ Wdc)
        return max(f, 0.0), M


class EulerPDController:
    """Small-angle cascaded controller: position PD -> roll/pitch setpoints -> Euler-angle PD."""
    name = "Linear small-angle (Euler PD)"

    def __init__(self, p=Params(), kx=6.0, kv=4.5, kR=225.0, kW=27.0):
        self.p, self.kx, self.kv, self.kR, self.kW = p, kx, kv, kR, kW

    def __call__(self, q, ref):
        m, J = self.p.m, self.p.J
        a = ref["a"] - self.kx * (q.x - ref["x"]) - self.kv * (q.v - ref["v"])
        psi = ref["yaw"]
        phi_d = (a[0] * np.sin(psi) - a[1] * np.cos(psi)) / G
        th_d = (a[0] * np.cos(psi) + a[1] * np.sin(psi)) / G
        R = q.R
        phi = np.arctan2(R[2, 1], R[2, 2])
        th = -np.arcsin(np.clip(R[2, 0], -1, 1))
        ps = np.arctan2(R[1, 0], R[0, 0])
        err = np.array([phi_d - phi, th_d - th, np.arctan2(np.sin(psi - ps), np.cos(psi - ps))])
        M = J @ (self.kR * err - self.kW * q.w)
        f = m * (G + a[2])          # hover linearisation: ignores tilt
        return max(f, 0.0), M
