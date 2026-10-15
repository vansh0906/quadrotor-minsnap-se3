"""Quadrotor experiments: minimum-snap time allocation, trajectory tracking at two speeds,
a wind gust, and recovery from an almost upside-down start.

python main.py            (about 1 minute; prints the tables and saves quadrotor.png)
"""
import argparse
import time

import numpy as np

import minsnap as ms
import quad as Q

WAYPOINTS = np.array([[0, 0, 1.0], [3, 0, 1.5], [4, 3, 2.5], [1, 4, 2.0], [-2, 2, 1.0], [0, 0, 1.0]])
DT, CTRL_EVERY = 1e-3, 2          # 1 kHz physics, 500 Hz control


def fly(ctrl, traj, wind_fn=None, R0=None, x0=None, hold=2.0):
    q = Q.Quadrotor(R0=R0, x0=traj.eval(0) if x0 is None else x0)
    T = traj.duration + hold
    n = int(T / DT)
    errs, tilts, log = [], [], []
    crashed = False
    for k in range(n):
        t = k * DT
        if k % CTRL_EVERY == 0:
            ref = Q.flat_reference(traj, t)
            f, M = ctrl(q, ref)
        wind = wind_fn(t) if wind_fn else np.zeros(3)
        q.step(f, M, DT, wind)
        e = np.linalg.norm(q.x - ref["x"])
        errs.append(e); tilts.append(q.tilt_deg())
        if k % 20 == 0:
            log.append((t, *q.x, *ref["x"], e))
        if e > 3.0 or not np.all(np.isfinite(q.x)):
            crashed = True
            break
    errs = np.array(errs)
    return dict(rms=np.sqrt(np.mean(errs ** 2)), max=errs.max(), tilt=max(tilts), crashed=crashed,
                sat=100.0 * q.saturated / max(1, len(errs)), log=np.array(log))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    print("1) Minimum-snap trajectory: 6 waypoints, 7th-order polynomials, continuity up to the 6th derivative\n")
    T_dist = ms.allocate_times(WAYPOINTS, 1.0)
    T_equal = np.full(len(T_dist), T_dist.mean())
    k_time = 50.0
    T_opt = ms.optimize_times(WAYPOINTS, T_dist, k_time)
    print(f"{'segment times':<26}{'total time':>11}{'snap cost':>12}{'J = snap + 50*T':>18}")
    rows = {}
    for name, T in (("equal", T_equal), ("proportional to distance", T_dist), ("optimised", T_opt)):
        tr = ms.Trajectory(WAYPOINTS, T)
        rows[name] = tr
        print(f"{name:<26}{tr.duration:>9.2f} s{tr.cost:>12.1f}{tr.cost + k_time * tr.duration:>18.1f}")

    traj_slow = ms.Trajectory(WAYPOINTS, ms.allocate_times(WAYPOINTS, 1.0))
    traj_fast = ms.Trajectory(WAYPOINTS, ms.allocate_times(WAYPOINTS, 2.5))
    gust = lambda t: np.array([3.0, -2.0, 0.0]) if 3.0 <= t < 5.0 else np.zeros(3)
    ctrls = [Q.GeometricController(), Q.GeometricController(feedforward=False), Q.EulerPDController()]

    print("\n2) Tracking (1 kg quadrotor, motors 0-6 N with 20 ms lag, 500 Hz control)\n")
    print(f"{'scenario':<34}{'controller':<33}{'RMS err':>9}{'max err':>9}{'max tilt':>10}")
    results = {}
    for scen, traj, wind in (("inspection loop, avg 1.0 m/s", traj_slow, None),
                             ("inspection loop, avg 2.5 m/s", traj_fast, None),
                             ("avg 2.5 m/s + 3.6 N wind gust", traj_fast, gust)):
        for c in ctrls:
            r = fly(c, traj, wind)
            results[(scen, c.name)] = r
            err = "CRASHED" if r["crashed"] else f"{r['rms'] * 100:6.1f} cm{r['max'] * 100:6.1f} cm"
            print(f"{scen:<34}{c.name:<33}{err:>18}{r['tilt']:>8.0f}°")
        print()

    print("3) Recovery: hover target at (0, 0, 2) m, start rolled 170° (almost upside down)\n")
    hover = ms.Trajectory(np.array([[0, 0, 2.0], [0, 0, 2.0001], [0, 0, 2.0]]), [2.0, 2.0])
    for c in [ctrls[0], Q.GeometricController(large_angle=False), ctrls[2]]:
        r = fly(c, hover, R0=Q.rot_x(np.radians(170)), x0=[0, 0, 2.0], hold=4.0)
        settle = f"did not recover (error passed {r['max']:.1f} m)" if r["crashed"] or r["log"][-1][-1] > 0.1 else \
            f"recovered: max error {r['max']:.2f} m, final error {r['log'][-1][-1] * 100:.1f} cm"
        print(f"   {c.name:<33}{settle}")

    print(f"\n(ran in {time.time() - t0:.0f} s)")
    if not a.no_plot:
        plot(traj_fast, results)


def plot(traj, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ts = np.linspace(0, traj.duration, 600)
    P = np.array([traj.eval(t) for t in ts])
    ax.plot(P[:, 0], P[:, 1], P[:, 2], "k--", lw=1, label="min-snap reference (2.5 m/s)")
    ax.scatter(*WAYPOINTS.T, color="k", s=20)
    cols = {"Geometric SE(3) + feed-forward": "tab:green", "Geometric SE(3), feedback only": "tab:blue",
            "Linear small-angle (Euler PD)": "tab:red"}
    for (scen, name), r in results.items():
        if scen == "inspection loop, avg 2.5 m/s":
            L = r["log"]
            ax.plot(L[:, 1], L[:, 2], L[:, 3], color=cols[name], lw=1.2, label=name)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Fast inspection loop")
    a2 = fig.add_subplot(1, 2, 2)
    for (scen, name), r in results.items():
        if scen == "avg 2.5 m/s + 3.6 N wind gust":
            L = r["log"]
            a2.plot(L[:, 0], L[:, 7] * 100, color=cols[name], lw=1.2, label=name)
    a2.axvspan(3, 5, color="tab:orange", alpha=0.15, label="wind gust")
    a2.set_xlabel("time (s)"); a2.set_ylabel("position error (cm)"); a2.set_title("Tracking error with a wind gust")
    a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("quadrotor.png", dpi=120)
    print("saved quadrotor.png")


if __name__ == "__main__":
    main()
