import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# --- Helper: skew-symmetric "hat" operator ---
def hat(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

# --- Digit pattern generator ---
def digit_pattern(digit, z=10, scale=1.0, offset=(0,0)):
    ox, oy = offset
    if digit == 1:
        pts = np.array([[ox, oy+i*scale, z] for i in range(10)])
    elif digit == 2:
        pts = np.array([[ox+i*scale, oy+9*scale, z] for i in range(4)] +
                       [[ox+3*scale, oy+9*scale-j*scale, z] for j in range(5)] +
                       [[ox+2-j*scale, oy, z] for j in range(4)])
    elif digit == 3:
        pts = np.array([[ox+i*scale, oy+9*scale, z] for i in range(4)] +
                       [[ox+3*scale, oy+5*scale, z]]*2 +
                       [[ox+3*scale, oy+j*scale, z] for j in range(5)] +
                       [[ox+i*scale, oy, z] for i in range(4)])
    elif digit == 4:
        pts = np.array([[ox, oy+9*scale-j*scale, z] for j in range(5)] +
                       [[ox+i*scale, oy+5*scale, z] for i in range(4)] +
                       [[ox+3*scale, oy+9*scale-j*scale, z] for j in range(10)])
    elif digit == 5:
        pts = np.array([[ox+i*scale, oy+9*scale, z] for i in range(4)] +
                       [[ox, oy+8*scale-j*scale, z] for j in range(5)] +
                       [[ox+i*scale, oy+4*scale, z] for i in range(4)] +
                       [[ox+3*scale, oy+3*scale-j*scale, z] for j in range(4)] +
                       [[ox+i*scale, oy, z] for i in range(4)])
    else:
        raise ValueError("Digit not supported")

    idx = np.linspace(0, len(pts)-1, 10, dtype=int)
    return pts[idx]

def generate_digit_sequences():
    patterns = [digit_pattern(d, z=10, scale=1.0, offset=(0,0)) for d in range(1,6)]
    trajectories = []
    for i in range(10):
        traj = []
        for pat in patterns:
            traj.append(pat[i])
        trajectories.append(np.array(traj))
    return trajectories

# --- Attack function ---
def attack(var, ic, type='control'):
    Su = np.diag([1.5, 1.5 , 1.0])
    Sx = np.linalg.inv(Su)
    du = 0
    dx = ic - (Sx @ ic)
    if type == 'obs':
        return (Sx @ var) + dx
    elif type == 'control':
        return (Su @ var) + du

# --- Drone class ---
class Drone:
    def __init__(self, trajectory, idx, color='b', attack_ctrl=False, attack_obs=False):
        self.m = 1.0
        self.g = 9.81
        self.J = np.diag([0.02, 0.02, 0.04])
        self.dt = 0.01

        self.Kp = 5
        self.Kd = 3.0
        self.Ki = 0
        self.K_R = 500
        self.K_w = 10
        self.K_avoid = 0
        self.safety_radius = 1.0
        self.integral_error = np.zeros((3,1))

        x_pos = idx * 1 + 0.25
        y_pos = idx * 1 + 0.25
        self.p = np.array([[x_pos],[y_pos],[0]])
        self.p_obs = self.p.copy()
        self.p_ic = self.p.copy()
        self.v = np.zeros((3,1))
        self.v_obs = self.v.copy()
        self.v_ic = self.v.copy()
        self.R = np.eye(3)
        self.omega = np.zeros((3,1))

        self.trajectory = trajectory
        self.stage = 0
        self.nextPoint = self.trajectory[self.stage,:].reshape(3,1)

        # histories
        self.posHistory = []       # actual position history
        self.posHistory_obs = []   # observed position history

        self.color = color
        self.attack_ctrl = attack_ctrl
        self.attack_obs = attack_obs

    def step(self, others):
        dt, m, g, J = self.dt, self.m, self.g, self.J
        p_des = self.nextPoint

        # Avoidance (kept; set K_avoid>0 to use)
        a_avoid = np.zeros((3,1))
        for other in others:
            if other is self: continue
            diff = self.p_obs - other.p_obs
            dist = np.linalg.norm(diff)
            if dist < self.safety_radius and dist > 1e-3:
                a_avoid += self.K_avoid * diff / (dist**2)

        # Outer loop (on obs state)
        self.integral_error += (p_des - self.p_obs) * dt
        a_cmd = self.Kp*(p_des - self.p_obs) - self.Kd*self.v_obs + self.Ki*self.integral_error + a_avoid
        if self.attack_ctrl:
            a_cmd = attack(a_cmd, ic=self.p_ic, type='control')

        # Orientation + dynamics
        a_cmd[2] += g
        f_des = m*a_cmd
        T = np.linalg.norm(f_des)
        if T < 1e-6: T = 1e-6
        b3_des = f_des / T
        psi_des = 0.0
        b1_psi = np.array([[np.cos(psi_des)], [np.sin(psi_des)], [0]])
        b2_des = np.cross(b3_des.flatten(), b1_psi.flatten())
        if np.linalg.norm(b2_des) < 1e-6:
            b2_des = np.array([0,1,0])
        else:
            b2_des /= np.linalg.norm(b2_des)
        b1_des = np.cross(b2_des, b3_des.flatten())
        R_des = np.column_stack((b1_des, b2_des, b3_des.flatten()))

        e_R_mat = 0.5*(R_des.T @ self.R - self.R.T @ R_des)
        e_R = np.array([[e_R_mat[2,1]], [e_R_mat[0,2]], [e_R_mat[1,0]]])
        tau = -self.K_R*e_R - self.K_w*self.omega + np.cross(self.omega.flatten(), (J @ self.omega).flatten()).reshape(3,1)

        omega_dot = np.linalg.inv(self.J) @ (tau - np.cross(self.omega.flatten(), (self.J @ self.omega).flatten()).reshape(3,1))
        self.omega += omega_dot*dt
        if np.linalg.norm(self.omega) > 50:
            self.omega = 50 * self.omega / np.linalg.norm(self.omega)

        R_update = expm(hat((self.omega*dt).flatten()))
        if np.any(np.isnan(R_update)) or np.any(np.isinf(R_update)):
            R_update = np.eye(3)
        self.R = self.R @ R_update
        u, _, vh = np.linalg.svd(self.R)
        self.R = u @ vh

        acc = np.array([[0],[0],[-self.g]]) + (T/self.m)*self.R[:,2].reshape(3,1)
        self.v += acc*dt
        self.p += self.v*dt

        # store actual position
        self.posHistory.append(self.p.flatten())

        # Observation attack
        if self.attack_obs:
            self.p_obs = attack(self.p, ic=self.p_ic, type='obs')
            self.v_obs = attack(self.v, ic=self.v_ic, type='obs')
        else:
            self.p_obs = self.p
            self.v_obs = self.v

        # store observed position
        self.posHistory_obs.append(self.p_obs.flatten())

        # stage change
        time_in_stage = len(self.posHistory_obs)*dt - self.stage*5
        if time_in_stage >= 5.0 and self.stage < len(self.trajectory)-1:
            self.stage += 1
            self.nextPoint = self.trajectory[self.stage,:].reshape(3,1)

# --- Collision checking ---
def check_collisions(drones, use_obs=True, radius=0.5):
    count = 0
    for i in range(len(drones)):
        for j in range(i+1, len(drones)):
            p1 = drones[i].p_obs if use_obs else drones[i].p
            p2 = drones[j].p_obs if use_obs else drones[j].p
            if np.linalg.norm(p1 - p2) < radius:
                count += 1
    return count

# --- Simulation ---
def simulation():
    Tsim = 25
    num_drones = 10
    attack_ctrl, attack_obs = True, True

    N = round(Tsim/0.01)
    trajectories = generate_digit_sequences()

    colors = plt.cm.tab10(np.linspace(0,1,num_drones))
    drones = [Drone(trajectories[i], i, color=colors[i], attack_ctrl=attack_ctrl, attack_obs=attack_obs) for i in range(num_drones)]
    drones_nominal = [Drone(trajectories[i], i, color=colors[i], attack_ctrl=False, attack_obs=False) for i in range(num_drones)]

    plt.ion()
    fig = plt.figure(figsize=(24,16))  # taller to accommodate 4 rows

    # Row 1: 3D animations
    ax_live_obs     = fig.add_subplot(4,3,1, projection='3d')
    ax_live_actual  = fig.add_subplot(4,3,2, projection='3d')
    ax_ref          = fig.add_subplot(4,3,3, projection='3d')  # nominal positions/animation

    # Row 2: Nominal positions vs time
    ax_nom_x = fig.add_subplot(4,3,4)
    ax_nom_y = fig.add_subplot(4,3,5)
    ax_nom_z = fig.add_subplot(4,3,6)

    # Row 3: Observed positions vs time
    ax_obs_x = fig.add_subplot(4,3,7)
    ax_obs_y = fig.add_subplot(4,3,8)
    ax_obs_z = fig.add_subplot(4,3,9)

    # Row 4: Actual positions vs time
    ax_act_x = fig.add_subplot(4,3,10)
    ax_act_y = fig.add_subplot(4,3,11)
    ax_act_z = fig.add_subplot(4,3,12)

    # Markers for top-row 3D plots
    for drone in drones:
        drone.marker_obs_top,   = ax_live_obs.plot([], [], [], marker='o', color=drone.color, markersize=6)
        drone.marker_actual_top,= ax_live_actual.plot([], [], [], marker='o', color=drone.color, markersize=6)
    for drone in drones_nominal:
        drone.marker_nominal_top, = ax_ref.plot([], [], [], marker='o', color=drone.color, markersize=6)

    # Collision + stage text
    text_obs = ax_live_obs.text2D(0.05, 0.95, "", transform=ax_live_obs.transAxes, fontsize=12)
    text_act = ax_live_actual.text2D(0.05, 0.95, "", transform=ax_live_actual.transAxes, fontsize=12)
    text_nom = ax_ref.text2D(0.05, 0.95, "", transform=ax_ref.transAxes, fontsize=12)

    # Axis bounds for 3D plots
    traj_arr = np.array(trajectories)
    extra = 5
    x_min, x_max = np.min(traj_arr[:,:,0])-extra, np.max(traj_arr[:,:,0])+extra
    y_min, y_max = np.min(traj_arr[:,:,1])-extra, np.max(traj_arr[:,:,1])+extra
    z_min, z_max = np.min(traj_arr[:,:,2])-extra, np.max(traj_arr[:,:,2])+extra
    for ax in [ax_live_obs, ax_live_actual, ax_ref]:
        ax.set_xlim([x_min, x_max]); ax.set_ylim([y_min, y_max]); ax.set_zlim([z_min, z_max])
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")

    ax_live_obs.set_title("Observed Drone Animation")
    ax_live_actual.set_title("Actual Drone Animation")
    ax_ref.set_title("Nominal Drone Animation")

    # Time-series lines
    nom_x_lines, nom_y_lines, nom_z_lines = [], [], []
    obs_x_lines, obs_y_lines, obs_z_lines = [], [], []
    act_x_lines, act_y_lines, act_z_lines = [], [], []

    col_obs_total = 0
    col_act_total = 0
    col_nom_total = 0

    for i in range(num_drones):
        c = colors[i]
        nom_x_lines.append(ax_nom_x.plot([], [], color=c)[0])
        nom_y_lines.append(ax_nom_y.plot([], [], color=c)[0])
        nom_z_lines.append(ax_nom_z.plot([], [], color=c)[0])
        obs_x_lines.append(ax_obs_x.plot([], [], color=c)[0])
        obs_y_lines.append(ax_obs_y.plot([], [], color=c)[0])
        obs_z_lines.append(ax_obs_z.plot([], [], color=c)[0])
        act_x_lines.append(ax_act_x.plot([], [], color=c)[0])
        act_y_lines.append(ax_act_y.plot([], [], color=c)[0])
        act_z_lines.append(ax_act_z.plot([], [], color=c)[0])

    # Labels
    for ax, lab in zip([ax_nom_x, ax_nom_y, ax_nom_z], ['X [m]', 'Y [m]', 'Z [m]']):
        ax.set_title(f"Nominal {lab.split()[0]}"); ax.set_ylabel(lab)
    for ax, lab in zip([ax_obs_x, ax_obs_y, ax_obs_z], ['X [m]', 'Y [m]', 'Z [m]']):
        ax.set_title(f"Observed {lab.split()[0]}"); ax.set_ylabel(lab)
    for ax, lab in zip([ax_act_x, ax_act_y, ax_act_z], ['X [m]', 'Y [m]', 'Z [m]']):
        ax.set_title(f"Actual {lab.split()[0]}"); ax.set_ylabel(lab)

    # Simulation loop
    for k in range(N):
        for drone in drones: drone.step(drones)
        for drone in drones_nominal: drone.step(drones_nominal)

        if k % 20 == 0:
            t_obs = np.arange(len(drones[0].posHistory_obs)) * drones[0].dt
            t_act = np.arange(len(drones[0].posHistory)) * drones[0].dt
            t_nom = np.arange(len(drones_nominal[0].posHistory)) * drones_nominal[0].dt

            # update 3D markers
            for d in drones:
                if d.posHistory_obs:
                    p_obs = d.posHistory_obs[-1]
                    d.marker_obs_top.set_data([p_obs[0]], [p_obs[1]])
                    d.marker_obs_top.set_3d_properties([p_obs[2]])
                if d.posHistory:
                    p_act = d.posHistory[-1]
                    d.marker_actual_top.set_data([p_act[0]], [p_act[1]])
                    d.marker_actual_top.set_3d_properties([p_act[2]])
            for dn in drones_nominal:
                if dn.posHistory:
                    p_nom = dn.posHistory[-1]
                    dn.marker_nominal_top.set_data([p_nom[0]], [p_nom[1]])
                    dn.marker_nominal_top.set_3d_properties([p_nom[2]])

            # check collisions at this step
            col_obs = check_collisions(drones, use_obs=True)
            col_act = check_collisions(drones, use_obs=False)
            col_nom = check_collisions(drones_nominal, use_obs=False)

            # accumulate totals
            col_obs_total += col_obs
            col_act_total += col_act
            col_nom_total += col_nom

            # stage info
            stage = drones[0].stage + 1
            sim_time = k * drones[0].dt

            # update the texts
            text_obs.set_text(f"t={sim_time:.1f}s | Collisions: {col_obs_total} | Stage: {stage}/5")
            text_act.set_text(f"t={sim_time:.1f}s | Collisions: {col_act_total} | Stage: {stage}/5")
            text_nom.set_text(f"t={sim_time:.1f}s | Collisions: {col_nom_total} | Stage: {stage}/5")

            # update time-series
            for i in range(num_drones):
                p_nom_hist = np.array(drones_nominal[i].posHistory)
                if len(p_nom_hist) > 0:
                    nom_x_lines[i].set_data(t_nom[:len(p_nom_hist)], p_nom_hist[:,0])
                    nom_y_lines[i].set_data(t_nom[:len(p_nom_hist)], p_nom_hist[:,1])
                    nom_z_lines[i].set_data(t_nom[:len(p_nom_hist)], p_nom_hist[:,2])
                p_obs_hist = np.array(drones[i].posHistory_obs)
                if len(p_obs_hist) > 0:
                    obs_x_lines[i].set_data(t_obs[:len(p_obs_hist)], p_obs_hist[:,0])
                    obs_y_lines[i].set_data(t_obs[:len(p_obs_hist)], p_obs_hist[:,1])
                    obs_z_lines[i].set_data(t_obs[:len(p_obs_hist)], p_obs_hist[:,2])
                p_act_hist = np.array(drones[i].posHistory)
                if len(p_act_hist) > 0:
                    act_x_lines[i].set_data(t_act[:len(p_act_hist)], p_act_hist[:,0])
                    act_y_lines[i].set_data(t_act[:len(p_act_hist)], p_act_hist[:,1])
                    act_z_lines[i].set_data(t_act[:len(p_act_hist)], p_act_hist[:,2])

            for ax in [ax_nom_x, ax_nom_y, ax_nom_z,
                       ax_obs_x, ax_obs_y, ax_obs_z,
                       ax_act_x, ax_act_y, ax_act_z]:
                ax.relim(); ax.autoscale_view()

            plt.pause(0.0001)

    plt.ioff(); plt.show()

if __name__ == "__main__":
    simulation()
