import numpy as np


def control_loop(
    rc,
    robot,
    trajectory,
    qdt_traj=None,
    taudt_threshold=None,
    disturbance_failure=None,
    measurement_failure=None,
    q_init=None,
    qdt_init=None,
    verbose=False
):
    """
    Run closed-loop control using the reservoir controller for a given trajectory.

    Args:
        rc: ReservoirComputing instance (stateful)
        robot: TwoLinkRobot instance (must have .dt)
        trajectory: ndarray [T, 2] of (x, y) points (desired trajectory)
        qdt_traj: ndarray [T, 2] of joint velocities (optional)
        taudt_threshold: np.array([-thr, +thr]) (optional, default [-5e-2, 5e-2])
        disturbance_failure: (2, T) array (optional, default zeros)
        measurement_failure: (4, T) array (optional, default zeros)
        q_init: array-like (2,) initial joint angles
        qdt_init: array-like (2,) initial joint velocities
        verbose: Print debug info
    Returns:
        data_pred: predicted (x, y) trajectory, ndarray [T, 2]
    """

    dt = robot.dt
    T = trajectory.shape[0]
    print("T =", T)

    if qdt_traj is None:
        qdt_traj = np.zeros((T, 2))
    if taudt_threshold is None:
        taudt_threshold = np.array([-5e-2, 5e-2])
    if disturbance_failure is None:
        disturbance_failure = np.zeros((2, T))
    if measurement_failure is None:
        measurement_failure = np.zeros((4, T))
    if q_init is None or qdt_init is None:
        raise ValueError("q_init and qdt_init must be provided and must match the trajectory's initial state.")

    q_pred = np.zeros((T+10, 2))
    qdt_pred = np.zeros((T+10, 2))
    q2dt_pred = np.zeros((T+10, 2))
    tau_pred = np.zeros((T+10, 2))
    data_pred = np.zeros((T+10, 2))

    q_pred[0, :] = q_init
    qdt_pred[0, :] = qdt_init

    # Build initial input u as in your closed-loop code
    u = np.hstack([
        trajectory[0, :],
        trajectory[1, :] if T > 1 else trajectory[0, :],
        qdt_traj[0, :],
        qdt_traj[1, :] if T > 1 else qdt_traj[0, :]
    ])

    for t_i in range(T - 3):
        u_batch = u.reshape(1, 1, -1)
        y_pred = rc.predict(u_batch)

        if y_pred.shape != (1, 1, 2):
            raise RuntimeError(f"Expected prediction shape (1,1,2), got {y_pred.shape}")

        predict_value = y_pred[0, 0]

        predict_value = predict_value.copy()
        predict_value += predict_value * disturbance_failure[:, t_i]
        time_li = 0 if t_i == 0 else t_i - 1
        for li in range(2):
            diff = predict_value[li] - tau_pred[time_li, li]
            if diff > taudt_threshold[1] * dt:
                predict_value[li] = tau_pred[time_li, li] + taudt_threshold[1] * dt
            if diff < taudt_threshold[0] * dt:
                predict_value[li] = tau_pred[time_li, li] + taudt_threshold[0] * dt

        tau_pred[t_i, :] = predict_value
        time_now = t_i

        q2dt_pred[time_now, :] = robot.forward_dynamics(
            q_pred[time_now, :], qdt_pred[time_now, :], predict_value
        )

        q_pred[time_now + 1, :] = q_pred[time_now, :] + qdt_pred[time_now, :] * dt
        qdt_pred[time_now + 1, :] = qdt_pred[time_now, :] + q2dt_pred[time_now, :] * dt

        x_pred, y_pred = robot.forward_kinematics(q_pred[time_now + 1, 0], q_pred[time_now + 1, 1])
        x_pred_measurement = x_pred + x_pred * measurement_failure[0, t_i]
        y_pred_measurement = y_pred + y_pred * measurement_failure[1, t_i]
        qdt_measurement_f_value = qdt_pred[time_now + 1, :] * measurement_failure[2:4, t_i]
        qdt_pred_measurement = qdt_pred[time_now + 1, :] + qdt_measurement_f_value
        data_pred[time_now + 1, :] = [x_pred, y_pred]

        u[0:2] = [x_pred_measurement, y_pred_measurement]
        u[2:4] = trajectory[time_now + 2, :] if time_now + 2 < T else trajectory[-1, :]
        u[4:6] = qdt_pred_measurement
        u[6:8] = qdt_traj[time_now + 2, :] if time_now + 2 < T else qdt_traj[-1, :]

        if verbose and t_i == 0:
            print("First iter:", predict_value, q_pred[1, :], qdt_pred[1, :])
        if verbose and t_i == T - 4:
            print("Last iter:", predict_value, q_pred[time_now + 1, :], qdt_pred[time_now + 1, :])

    return data_pred[:T, :]
