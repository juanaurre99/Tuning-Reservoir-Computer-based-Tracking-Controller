import numpy as np
from scipy.ndimage import gaussian_filter1d

class TwoLinkRobot:
    def __init__(self, m1, m2, l1, l2, lc1, lc2, I1, I2, dt=0.01, traj_frequency=1.0):
        # Robot parameters
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.lc1 = lc1
        self.lc2 = lc2
        self.I1 = I1
        self.I2 = I2
        # Trajectory/Simulation parameters
        self.dt = dt
        self.traj_frequency = traj_frequency

    # ---- Kinematics and Dynamics ----

    def compute_torques(self, q, qd, qdd):
        T = q.shape[0]
        tau = np.zeros((T, 2))
        for i in range(T):
            q1, q2 = q[i]
            dq1, dq2 = qd[i]
            ddq1, ddq2 = qdd[i]

            H11 = self.m1*self.lc1**2 + self.I1 + self.m2*(self.l1**2 + self.lc2**2 + 2*self.l1*self.lc2*np.cos(q2)) + self.I2
            H12 = self.m2*self.l1*self.lc2*np.cos(q2) + self.m2*self.lc2**2 + self.I2
            H21 = H12
            H22 = self.m2*self.lc2**2 + self.I2
            h = self.m2*self.l1*self.lc2*np.sin(q2)

            c1 = -h*dq2*dq1 - h*(dq1 + dq2)*dq2
            c2 = h*dq1**2

            tau[i, 0] = H11*ddq1 + H12*ddq2 + c1
            tau[i, 1] = H21*ddq1 + H22*ddq2 + c2
        return tau

    def forward_dynamics(self, q, qd, tau):
        q1, q2 = q
        dq1, dq2 = qd
        tau1, tau2 = tau

        H11 = self.m1*self.lc1**2 + self.I1 + self.m2*(self.l1**2 + self.lc2**2 + 2*self.l1*self.lc2*np.cos(q2)) + self.I2
        H12 = self.m2*self.l1*self.lc2*np.cos(q2) + self.m2*self.lc2**2 + self.I2
        H21 = H12
        H22 = self.m2*self.lc2**2 + self.I2
        h = self.m2*self.l1*self.lc2*np.sin(q2)

        C1 = -h * dq2 * dq1 - h * (dq1 + dq2) * dq2
        C2 = h * dq1**2

        H = np.array([[H11, H12], [H21, H22]])
        C = np.array([C1, C2])
        T = np.array([tau1, tau2])

        qdd = np.linalg.solve(H, T - C)
        return qdd

    def forward_kinematics(self, q1, q2):
        x = self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2)
        y = self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2)
        return x, y

    def inverse_kinematics(self, x, y, initial_q=None):
        """
        Compute joint angles (q1, q2) for desired end-effector path (x, y).

        If initial_q is provided, checks and corrects for the correct elbow-up/down branch.
        """
        T = len(x)
        q2 = np.arccos((x ** 2 + y ** 2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2))

        # Branch flipping logic (from MATLAB)
        symb = 1
        for i in range(1, T - 1):
            q2[i] = symb * q2[i]
            if np.isclose(np.abs(q2[i]), np.pi) or np.isclose(q2[i], 0.0):
                dx1 = x[i + 1] - x[i]
                dx0 = x[i] - x[i - 1]
                if np.sign(dx1) == np.sign(dx0):
                    symb = -symb

        # First attempt: q1 as usual
        q1 = np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2))

        # Check initial pose match; if not, flip branch as in MATLAB
        if initial_q is not None:
            x1 = self.l1 * np.cos(q1[0])
            x1_expected = self.l1 * np.cos(initial_q[0])
            if np.round(x1, 2) != np.round(x1_expected, 2):
                q2 = -q2
                q1 = np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2))

                # Quadrant corrections for q1 (MATLAB-style)
                for ij in range(len(q1)):
                    if x[ij] < 0 and y[ij] > 0:
                        q1[ij] = q1[ij] + np.pi
                    elif x[ij] < 0 and y[ij] < 0:
                        q1[ij] = q1[ij] + np.pi
                    elif x[ij] > 0 and y[ij] < 0:
                        q1[ij] = q1[ij] + 2 * np.pi

                # Jump/unwrapping corrections for q1/q2
                for ij in range(1, len(q1)):
                    dq1 = q1[ij] - q1[ij - 1]
                    dq2 = q2[ij] - q2[ij - 1]
                    # π jump
                    if np.pi - 0.1 < dq1 < np.pi + 0.1:
                        q1[ij] = q1[ij] - np.pi
                    if -np.pi - 0.1 < dq1 < -np.pi + 0.1:
                        q1[ij] = q1[ij] + np.pi
                    # 2π unwrapping
                    if dq1 > np.pi:
                        q1[ij:] = q1[ij:] - 2 * np.pi
                    if dq1 < -np.pi:
                        q1[ij:] = q1[ij:] + 2 * np.pi
                    if dq2 > np.pi:
                        q2[ij:] = q2[ij:] - 2 * np.pi
                    if dq2 < -np.pi:
                        q2[ij:] = q2[ij:] + 2 * np.pi

        # Unwrap for smoothness (optional, since above is robust)
        q1 = np.unwrap(q1)
        q2 = np.unwrap(q2)

        return q1, q2

    # ---- Trajectory Generation ----

    def generate_trajectory(self, traj_type, val_length, q_init=None, bridge_type='cubic'):
        # Trajectory selector
        if traj_type == 'circle':
            x, y = self._circle(val_length)
        elif traj_type == 'infty':
            x, y = self._infty(val_length)
        elif traj_type == 'astroid':
            x, y = self._astroid(val_length)
        # Add more elifs for other types as needed
        elif traj_type == 'lorenz':
            x, y = self._lorenz(val_length)
        else:
            raise ValueError(f"Unknown trajectory type: {traj_type}")

        # Optional bridging from current config to trajectory
        if q_init is not None:
            x, y = self._bridge_to_trajectory(x, y, q_init, bridge_type=bridge_type)

        return x, y

    def _circle(self, val_length):
        t = np.arange(0, 2*val_length+1) * self.dt
        x = 0.5 * np.cos(2 * np.pi * t / self.traj_frequency)
        y = 0.5 * np.sin(2 * np.pi * t / self.traj_frequency)
        return x, y

    def _infty(self, val_length):
        t = np.arange(0, 2*val_length+1) * self.dt
        x = 0.25 * np.sin(2 * np.pi * t / (2 * self.traj_frequency))
        y = 0.15 * np.sin(2 * np.pi * t / self.traj_frequency)
        return x, y

    def _astroid(self, val_length):
        t = np.arange(0, 2*val_length+1) * self.dt
        x = 0.4 * (np.cos(2 * np.pi * t / 250))**3
        y = 0.4 * (np.sin(2 * np.pi * t / 250))**3
        return x, y

    def _lorenz(self, val_length, q_init=None):
        # Lorenz system parameters
        sigma = 10.0
        beta = 8.0 / 3.0
        rho = 28.0

        dt = self.dt
        T = 2 * val_length + 1
        xs = np.empty(T)
        ys = np.empty(T)
        zs = np.empty(T)
        # Initial conditions
        xs[0], ys[0], zs[0] = 0., 1., 1.05

        for i in range(T - 1):
            xs[i + 1] = xs[i] + sigma * (ys[i] - xs[i]) * dt
            ys[i + 1] = ys[i] + (xs[i] * (rho - zs[i]) - ys[i]) * dt
            zs[i + 1] = zs[i] + (xs[i] * ys[i] - beta * zs[i]) * dt

        # Normalize x/y for robot workspace (center, scale to safe radius)
        x = xs
        y = ys
        x = (x - np.mean(x)) / (np.max(np.abs(x)) + 1e-9) * 0.5
        y = (y - np.mean(y)) / (np.max(np.abs(y)) + 1e-9) * 0.5

        return x, y

    def _bridge_to_trajectory(self, x, y, q_init, bridge_type='cubic'):
        # Start EE position from q_init
        x_start = self.l1 * np.cos(q_init[0]) + self.l2 * np.cos(q_init[0] + q_init[1])
        y_start = self.l1 * np.sin(q_init[0]) + self.l2 * np.sin(q_init[0] + q_init[1])
        dists = np.sqrt((x - x_start) ** 2 + (y - y_start) ** 2)
        add_id = np.argmin(dists)

        # IK for nearest and next point (for velocity)
        q1_bg = []
        q2_bg = []
        for i in [add_id, add_id + 1]:
            q2_i = np.arccos((x[i] ** 2 + y[i] ** 2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2))
            q1_i = np.arctan2(y[i], x[i]) - np.arctan2(self.l2 * np.sin(q2_i), self.l1 + self.l2 * np.cos(q2_i))
            # Quadrant correction
            if x[i] < 0 and y[i] > 0 or x[i] < 0 and y[i] < 0:
                q1_i += np.pi
            elif x[i] > 0 and y[i] < 0:
                q1_i += 2 * np.pi
            q1_bg.append(q1_i)
            q2_bg.append(q2_i)
        q1_bg = np.array(q1_bg)
        q2_bg = np.array(q2_bg)

        # Initial joint and velocity (assume at rest or supply)
        theta0 = np.array(q_init)
        theta0_dot = np.array([0.0, 0.0])  # Or pass as argument
        theta1 = np.array([q1_bg[0], q2_bg[0]])
        theta1_dot = (np.array([q1_bg[1], q2_bg[1]]) - theta1) / self.dt

        bridge_len = np.linalg.norm([x[add_id] - x_start, y[add_id] - y_start])
        bridge_time = max(int(round(bridge_len / self.dt)), 2)
        t_bg = np.linspace(0, bridge_time * self.dt, bridge_time + 1)

        # Cubic coefficients (as in MATLAB)
        a0 = theta0
        a1 = theta0_dot
        a2 = 3 * (theta1 - theta0) / (bridge_time * self.dt) ** 2 - 2 * theta0_dot / (
                    bridge_time * self.dt) - theta1_dot / (bridge_time * self.dt)
        a3 = -2 * (theta1 - theta0) / (bridge_time * self.dt) ** 3 + (theta1_dot + theta0_dot) / (
                    bridge_time * self.dt) ** 2

        q1_bridge = a0[0] + a1[0] * t_bg + a2[0] * t_bg ** 2 + a3[0] * t_bg ** 3
        q2_bridge = a0[1] + a1[1] * t_bg + a2[1] * t_bg ** 2 + a3[1] * t_bg ** 3

        x_bridge = self.l1 * np.cos(q1_bridge) + self.l2 * np.cos(q1_bridge + q2_bridge)
        y_bridge = self.l1 * np.sin(q1_bridge) + self.l2 * np.sin(q1_bridge + q2_bridge)

        # Concatenate: skip 1st and last to avoid overlap with trajectory
        x_full = np.concatenate([x_bridge[1:-1], x[add_id:]])
        y_full = np.concatenate([y_bridge[1:-1], y[add_id:]])
        return x_full, y_full

    def generate_training_data(self, time_info, noise_level, smoothing_window=5):
        section_len = int(time_info['section_len'])
        time_length = int(time_info['time_length'])

        # Generate random torques
        tau = -noise_level + 2 * noise_level * np.random.rand(time_length, 2)

        # --- Apply moving average smoothing ---
        # Window must be at least 1 and not larger than time_length
        win = max(1, min(smoothing_window, time_length))
        if win > 1:
            kernel = np.ones(win) / win
            tau_smoothed = np.vstack([
                np.convolve(tau[:, 0], kernel, mode="same"),
                np.convolve(tau[:, 1], kernel, mode="same")
            ]).T
        else:
            tau_smoothed = tau

        q = np.zeros((time_length, 2))
        qd = np.zeros((time_length, 2))
        qdd = np.zeros((time_length, 2))

        # Random initial joint positions
        q[0, 0] = np.random.uniform(-2 * np.pi, 2 * np.pi)
        q[0, 1] = np.random.uniform(-2 * np.pi, 2 * np.pi)
        qd[0, :] = 0.0

        for t in range(time_length - 1):
            if t % section_len == 0 and t != 0:
                q[t, 0] = np.random.uniform(-2 * np.pi, 2 * np.pi)
                q[t, 1] = np.random.uniform(-2 * np.pi, 2 * np.pi)
                qd[t, :] = 0.0

            qdd[t] = self.forward_dynamics(q[t], qd[t], tau_smoothed[t])

            if t % section_len == 0 and t != 0:
                qdd[t, :] = 0.0
                tau_smoothed[t, :] = 0.0

            q[t + 1] = q[t] + qd[t] * self.dt
            qd[t + 1] = qd[t] + qdd[t] * self.dt

        x, y = self.forward_kinematics(q[:, 0], q[:, 1])
        xy = np.stack([x, y], axis=1)
        return xy, q, qd, qdd, tau_smoothed



    def generate_inverse_model_training_data(self, time_info, noise_level):
        """
        Generate input-output data for training a model of inverse dynamics.
        Input is [Cx(t), Cy(t), qd1(t), qd2(t), Cx(t+1), Cy(t+1), qd1(t+1), qd2(t+1)]
        Output is the torque tau(t) that caused the transition.

        Returns:
            X (ndarray): [T-1, 8]
            Y (ndarray): [T-1, 2]
        """
        xy, q, qd, qdd, tau = self.generate_training_data(time_info, noise_level)
        Cx, Cy = xy[:, 0], xy[:, 1]
        qd1, qd2 = qd[:, 0], qd[:, 1]
        # Stack current and next states
        y_t = np.stack([Cx[:-1], Cy[:-1], qd1[:-1], qd2[:-1]], axis=1)
        y_tp1 = np.stack([Cx[1:], Cy[1:], qd1[1:], qd2[1:]], axis=1)
        X = np.hstack([y_t, y_tp1])
        Y = tau[:-1]
        return X, Y

def moving_average(arr, window_size):
    kernel = np.ones(window_size) / window_size
    return np.array([
        np.convolve(arr[:, i], kernel, mode='same')
        for i in range(arr.shape[1])
    ]).T  #