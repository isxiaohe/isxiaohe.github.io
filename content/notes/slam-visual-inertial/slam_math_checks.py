"""Numerical checks accompanying the visual SLAM and inertial navigation notes.

Run: python3 slam_math_checks.py
Requires: Python 3 and NumPy. No dataset, ROS, GPU, or network is needed.
Do not use python -O: the assertions are part of the numerical checks.

Conventions: R_AB maps B coordinates into A coordinates; right rotation
perturbations; world-coordinate additive position errors. The SO(3) log
helper deliberately excludes rotations close to pi and is not a full
production rotation library. Preintegration checks compare the same
first-order discrete model, not arbitrary continuous-time trajectories.
"""
import json
import numpy as np


def hat(a):
    x, y, z = a
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def exp_so3(phi):
    theta = np.linalg.norm(phi)
    A = hat(phi)
    if theta < 1e-6:
        t2 = theta * theta
        s1 = 1. - t2 / 6. + t2 * t2 / 120.
        s2 = .5 - t2 / 24. + t2 * t2 / 720.
    else:
        s1 = np.sin(theta) / theta
        s2 = (1. - np.cos(theta)) / (theta * theta)
    return np.eye(3) + s1 * A + s2 * (A @ A)


def log_so3(R):
    v = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]]) * .5
    sn = np.linalg.norm(v)
    cs = (np.trace(R) - 1.) * .5
    theta = np.arctan2(sn, cs)
    if theta < 1e-8:
        return v * (1. + theta * theta / 6.)
    if sn < 1e-10:
        raise ValueError("This small checking helper deliberately excludes pi rotations")
    return theta * v / sn


def right_jacobian(phi):
    theta = np.linalg.norm(phi)
    A = hat(phi)
    if theta < 1e-6:
        t2 = theta * theta
        a = .5 - t2 / 24. + t2 * t2 / 720.
        b = 1. / 6. - t2 / 120. + t2 * t2 / 5040.
    else:
        a = (1. - np.cos(theta)) / theta**2
        b = (theta - np.sin(theta)) / theta**3
    return np.eye(3) - a * A + b * A @ A


def finite_difference(fun, dim, eps=1e-7):
    out = []
    for k in range(dim):
        step = np.zeros(dim)
        step[k] = eps
        out.append((fun(step) - fun(-step)) / (2. * eps))
    return np.column_stack(out)


def project(q):
    return np.array([520. * q[0] / q[2] + 320., 500. * q[1] / q[2] + 240.])


def projection_jacobian(q):
    x, y, z = q
    return np.array([[520./z, 0., -520.*x/z**2], [0., 500./z, -500.*y/z**2]])


def check_rotation():
    phi = np.array([.8, -.4, .25])
    R = exp_so3(phi)
    numerical = finite_difference(lambda d: log_so3(R.T @ exp_so3(phi+d)), 3)
    delta = np.array([.03, -.01, .02])
    return {
        "right_jacobian_max_abs": float(np.max(np.abs(numerical-right_jacobian(phi)))),
        "left_right_perturbation_max_abs": float(np.max(np.abs(R @ exp_so3(delta)-exp_so3(R@delta)@R))),
        "rotation_orthogonality_max_abs": float(np.max(np.abs(R.T@R-np.eye(3)))),
    }


def check_projection():
    R = exp_so3(np.array([.2, -.3, .15]))
    p = np.array([.3, -.2, 1.])
    P = p + R @ np.array([.5, -.4, 4.])
    obs = np.array([380., 193.])
    q = R.T@(P-p)
    Jpi = projection_jacobian(q)
    analytical = np.column_stack([-Jpi@hat(q), Jpi@R.T, -Jpi@R.T])
    def residual(d):
        qd = (R@exp_so3(d[:3])).T @ (P+d[6:9]-(p+d[3:6]))
        return obs-project(qd)
    numerical = finite_difference(residual, 9)

    Rbc = exp_so3(np.array([.1, .2, -.1]))
    pbc = np.array([.12, -.03, .06])
    qb = R.T@(P-p)
    qc = Rbc.T@(qb-pbc)
    Jc = projection_jacobian(qc)
    Jbody = np.column_stack([-Jc@Rbc.T@hat(qb), Jc@Rbc.T@R.T])
    def body_residual(d):
        qbd = (R@exp_so3(d[:3])).T @ (P-(p+d[3:6]))
        return obs-project(Rbc.T@(qbd-pbc))
    Jbody_num = finite_difference(body_residual, 6)
    return {
        "camera_pose_and_landmark_max_abs": float(np.max(np.abs(numerical-analytical))),
        "body_pose_with_extrinsics_max_abs": float(np.max(np.abs(Jbody_num-Jbody))),
    }


def check_gaussian_elimination(rng):
    A = rng.normal(size=(15, 7))
    H = A.T@A + .3*np.eye(7)
    eta = rng.normal(size=7)
    full = np.linalg.solve(H, eta)
    Haa, Hab, Hba, Hbb = H[:3,:3], H[:3,3:], H[3:,:3], H[3:,3:]
    S = Haa-Hab@np.linalg.solve(Hbb,Hba)
    eta_s = eta[:3]-Hab@np.linalg.solve(Hbb,eta[3:])
    aa = np.linalg.solve(S,eta_s)
    bb = np.linalg.solve(Hbb,eta[3:]-Hba@aa)

    Jx = rng.normal(size=(10,4))
    Jf = rng.normal(size=(10,3))
    Q, _ = np.linalg.qr(Jf,mode="complete")
    Q2 = Q[:,3:]
    Sqr = (Q2.T@Jx).T@(Q2.T@Jx)
    Ss = Jx.T@Jx - Jx.T@Jf@np.linalg.solve(Jf.T@Jf,Jf.T@Jx)

    M = rng.normal(size=(5,5))
    P = M@M.T + .5*np.eye(5)
    N = rng.normal(size=(3,3))
    V = N@N.T + .1*np.eye(3)
    C = rng.normal(size=(3,5))
    mu = rng.normal(size=5)
    z = rng.normal(size=3)
    Pi, Vi = np.linalg.inv(P), np.linalg.inv(V)
    Ppost = np.linalg.inv(Pi+C.T@Vi@C)
    mu_map = np.linalg.solve(Pi+C.T@Vi@C,Pi@mu+C.T@Vi@z)
    K = np.linalg.solve(C@P@C.T+V,C@P).T
    mu_kf = mu+K@(z-C@mu)
    Pkf = (np.eye(5)-K@C)@P@(np.eye(5)-K@C).T+K@V@K.T
    return {
        "full_vs_schur_max_abs": float(np.max(np.abs(full-np.r_[aa,bb]))),
        "nullspace_vs_schur_hessian_max_abs": float(np.max(np.abs(Sqr-Ss))),
        "map_vs_kalman_mean_max_abs": float(np.max(np.abs(mu_map-mu_kf))),
        "map_vs_kalman_covariance_max_abs": float(np.max(np.abs(Ppost-Pkf))),
    }


def check_gauge():
    R = exp_so3(np.array([.2, -.1, .3]))
    p = np.array([.4,.1,-.2])
    P = p+R@np.array([.4,.2,4.])
    Q = exp_so3(np.array([.5,-.3,.2]))
    shift = np.array([1.,2.,-1.])
    scale = 2.3
    Rp, pp, Pp = Q@R, scale*Q@p+shift, scale*Q@P+shift
    mono_error = np.max(np.abs(project(R.T@(P-p))-project(Rp.T@(Pp-pp))))

    Qyaw = exp_so3(np.array([0.,0.,.8]))
    g = np.array([0.,0.,-9.81])
    aw = np.array([.4,-.2,.3])
    f = R.T@(aw-g)
    fp = (Qyaw@R).T@(Qyaw@aw-g)
    fs = (Qyaw@R).T@(scale*Qyaw@aw-g)
    f_static = R.T@(-g)
    fs_static = (Qyaw@R).T@(scale*np.zeros(3)-g)
    return {
        "monocular_sim3_pixel_error": float(mono_error),
        "vio_yaw_accelerometer_error": float(np.max(np.abs(f-fp))),
        "excited_scale_changes_accelerometer_norm": float(np.linalg.norm(f-fs)),
        "constant_velocity_scale_accelerometer_error": float(np.max(np.abs(f_static-fs_static))),
    }


def integrate(world_R, world_v, world_p, g, samples):
    R, v, p = world_R.copy(), world_v.copy(), world_p.copy()
    for omega, force, dt in samples:
        aw = R@force+g
        p = p+v*dt+.5*aw*dt**2
        v = v+aw*dt
        R = R@exp_so3(omega*dt)
    return R,v,p


def check_inertial(rng):
    g = np.array([0.,0.,-9.81])
    R0 = exp_so3(np.array([.3,-.2,.4]))
    v0 = np.array([.2,-.1,.3])
    p0 = np.array([1.,-.2,.5])
    samples = []
    for _ in range(200):
        samples.append((rng.normal(scale=.3,size=3), rng.normal(scale=1.,size=3)+np.array([0.,0.,9.81]), .005))
    R,v,p = integrate(R0,v0,p0,g,samples)
    DR,Dv,Dp = integrate(np.eye(3),np.zeros(3),np.zeros(3),np.zeros(3),samples)
    total = sum(s[2] for s in samples)
    Rpred = R0@DR
    vpred = v0+g*total+R0@Dv
    ppred = p0+v0*total+.5*g*total**2+R0@Dp

    static_samples = [(np.zeros(3),-R0.T@g,.01)]*1000
    Rs,vs,ps = integrate(R0,np.zeros(3),p0,g,static_samples)
    _,vc,pc = integrate(R0,v0,p0,g,static_samples)
    return {
        "preintegration_rotation_max_abs": float(np.max(np.abs(R-Rpred))),
        "preintegration_velocity_max_abs": float(np.max(np.abs(v-vpred))),
        "preintegration_position_max_abs": float(np.max(np.abs(p-ppred))),
        "static_drift_position_norm_after_10s": float(np.linalg.norm(ps-p0)),
        "constant_velocity_position_error_after_10s": float(np.linalg.norm(pc-(p0+10.*v0))),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(711)
    results = {
        "1_rotation": check_rotation(),
        "2_projection": check_projection(),
        "3_gaussian_elimination": check_gaussian_elimination(rng),
        "4_gauge": check_gauge(),
        "5_inertial": check_inertial(rng),
    }
    limits = {
        "1_rotation": 1e-7,
        "2_projection": 3e-6,
        "3_gaussian_elimination": 1e-10,
        "5_inertial": 1e-9,
    }
    for group, limit in limits.items():
        for name, value in results[group].items():
            assert value < limit, (group,name,value,limit)
    for name, value in results["4_gauge"].items():
        if name == "excited_scale_changes_accelerometer_norm":
            assert value > .1
        else:
            assert value < 1e-9, (name,value)
    print(json.dumps(results,ensure_ascii=False,indent=2))
