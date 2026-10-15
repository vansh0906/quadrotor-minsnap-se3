"""Quick self-checks. Run: python tests.py"""
import numpy as np

import minsnap as ms
import quad as Q

wp = np.array([[0, 0, 1.0], [2, 0, 1.5], [3, 2, 2.0], [0, 0, 1.0]])
tr = ms.Trajectory(wp, ms.allocate_times(wp, 1.0))
for i, t in enumerate(tr.t_knots):                       # passes through every waypoint
    assert np.allclose(tr.eval(t), wp[i], atol=1e-6)
for k in (1, 2, 3):                                      # starts and ends at rest
    assert np.allclose(tr.eval(0, k), 0, atol=1e-6) and np.allclose(tr.eval(tr.duration, k), 0, atol=1e-6)
for t in tr.t_knots[1:-1]:                               # smooth up to snap at the joins
    for k in range(1, 5):
        assert np.allclose(tr.eval(t - 1e-7, k), tr.eval(t + 1e-7, k), atol=1e-4)
# the minimum-snap solution beats a perturbed feasible trajectory? check cost is minimal along a direction
T = ms.allocate_times(wp, 1.0)
assert ms.Trajectory(wp, T).cost < ms.Trajectory(wp, T * 0.8).cost      # faster = more snap
print("min-snap: waypoints, rest at the ends, continuity, cost scaling ... ok")

R = Q.expm_so3(np.array([0.3, -0.2, 0.5]))
assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1.0)
q = Q.Quadrotor()
assert np.allclose(q.A @ (q.A_inv @ np.array([9.81, 0.01, -0.02, 0.003])), [9.81, 0.01, -0.02, 0.003])
print("SO(3) exponential, motor allocation ... ok")

ref = dict(x=np.zeros(3), v=np.zeros(3), a=np.zeros(3), R=np.eye(3), w=np.zeros(3), wdot=np.zeros(3), yaw=0.0)
for ctrl in (Q.GeometricController(), Q.EulerPDController()):
    q = Q.Quadrotor(x0=[0.5, -0.4, 0.3])
    for k in range(4000):
        if k % 2 == 0:
            f, M = ctrl(q, ref)
        q.step(f, M, 1e-3)
    assert np.linalg.norm(q.x) < 0.01, ctrl.name
print("hover: both controllers return to the setpoint ... ok")
print("all checks passed")
