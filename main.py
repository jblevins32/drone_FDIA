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
def attack(u, ic, type='control'):
    Su = 0.75
    Sx = 1/Su
    du = 0
    dx = (ic * Sx) - ic
    if type == 'obs':
        return (u * Sx) + dx
    elif type == 'control':
        return (u * Su) + du

# --- Drone class ---
class Drone:
    def __init__(self, trajectory, idx, color='b', attack_ctrl=False, attack_obs=False):
        '''
        obs variables are those that are attacked with an observation attack. 
        If no obs, it is what the drone is actually doing.
        '''
        self.m = 1.0
        self.g = 9.81
        self.J = np.diag([0.02, 0.02, 0.04])
        self.dt = 0.01

        self.Kp = 5
        self.Kd = 4.0
        self.Ki = 1
        self.K_R = 500
        self.K_w = 20
        self.K_avoid = 10   # avoidance gain
        self.safety_radius = 1.0
        self.integral_error = np.zeros((3,1))

        # rand_x_pos = np.random.uniform(-5, 5)
        # rand_y_pos = np.random.uniform(-5, 5)
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

        self.posHistory_obs = []
        self.errHistory_obs = []           
        self.errHistory_pre_obs_attack = [] 

        self.color = color

        self.attack_ctrl = attack_ctrl
        self.attack_obs = attack_obs

    def step(self, others):
        dt, m, g, J = self.dt, self.m, self.g, self.J
        p_des = self.nextPoint

        # --- Avoidance ---
        a_avoid = np.zeros((3,1))
        for other in others:
            if other is self:
                continue
            diff = self.p_obs - other.p_obs
            dist = np.linalg.norm(diff)
            if dist < self.safety_radius and dist > 1e-3:
                a_avoid += self.K_avoid * diff / (dist**2)

        # --- Outer loop (controller acts on obs state) ---
        self.integral_error += (p_des - self.p_obs) * dt
        a_cmd = self.Kp*(p_des - self.p_obs) - self.Kd*self.v_obs + self.Ki*self.integral_error + a_avoid
        a_cmd[2] += g

        if self.attack_ctrl:
            a_cmd = attack(a_cmd, ic=self.p_ic, type='control')

        # --- Orientation ---
        f_des = m*a_cmd
        T = np.linalg.norm(f_des)
        if T < 1e-6: T = 1e-6
        b3_des = f_des / T
        psi_des = 0.0
        b1_psi = np.array([[np.cos(psi_des)], [np.sin(psi_des)], [0]])
        b2_des = np.cross(b3_des.flatten(), b1_psi.flatten())
        norm_b2 = np.linalg.norm(b2_des)
        if norm_b2 < 1e-6:
            b2_des = np.array([0,1,0])
        else:
            b2_des /= norm_b2
        b1_des = np.cross(b2_des, b3_des.flatten())
        R_des = np.column_stack((b1_des, b2_des, b3_des.flatten()))

        # --- Inner loop ---
        e_R_mat = 0.5*(R_des.T @ self.R - self.R.T @ R_des)
        e_R = np.array([[e_R_mat[2,1]], [e_R_mat[0,2]], [e_R_mat[1,0]]])
        tau = -self.K_R*e_R - self.K_w*self.omega + \
              np.cross(self.omega.flatten(), (J @ self.omega).flatten()).reshape(3,1)

        omega_dot = np.linalg.inv(J) @ (tau - np.cross(self.omega.flatten(), (J @ self.omega).flatten()).reshape(3,1))
        self.omega += omega_dot*dt
        if np.linalg.norm(self.omega) > 50:
            self.omega = 50 * self.omega / np.linalg.norm(self.omega)

        R_update = expm(hat((self.omega*dt).flatten()))
        if np.any(np.isnan(R_update)) or np.any(np.isinf(R_update)):
            R_update = np.eye(3)
        self.R = self.R @ R_update
        u, _, vh = np.linalg.svd(self.R)
        self.R = u @ vh

        acc = np.array([[0],[0],[-g]]) + (T/m)*self.R[:,2].reshape(3,1)
        self.v += acc*dt
        self.p += self.v*dt

        # --- Pre-attack error ---
        err_pre_obs_attack = np.linalg.norm(p_des - self.p)
        self.errHistory_pre_obs_attack.append(err_pre_obs_attack)

        # --- Observation attack ---
        if self.attack_obs:
            self.p_obs = attack(self.p, ic=self.p_ic, type='obs')
            self.v_obs = attack(self.v, ic=self.v_ic, type='obs')
        else: 
            self.p_obs = self.p
            self.v_obs = self.v

        # --- Post-attack error ---
        self.posHistory_obs.append(self.p_obs.flatten())
        self.errHistory_obs.append(np.linalg.norm(p_des - self.p_obs))

        # --- Waypoint switching ---
        time_in_stage = len(self.posHistory_obs)*dt - self.stage*5
        if time_in_stage >= 5.0 and self.stage < len(self.trajectory)-1:
            self.stage += 1
            self.nextPoint = self.trajectory[self.stage,:].reshape(3,1)

# --- Collision checking ---
def check_collisions(drones, use_obs=True, radius=0.00001):
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

    # Simulation parameters
    Tsim = 25
    num_drones = 1
    attack_ctrl = True
    attack_obs = True

    N = round(Tsim/0.01)
    trajectories = generate_digit_sequences()
    
    colors = plt.cm.tab10(np.linspace(0,1,10))
    drones = [Drone(trajectories[i], i, color=colors[i], attack_ctrl=attack_ctrl, attack_obs=attack_obs) for i in range(num_drones)]
    drones_nominal = [Drone(trajectories[i], i, color=colors[i], attack_ctrl=False, attack_obs=False) for i in range(num_drones)]

    plt.ion()
    fig = plt.figure(figsize=(24,6))
    ax_live_obs = fig.add_subplot(231, projection='3d')
    ax_live_actual = fig.add_subplot(232, projection='3d')
    ax_ref  = fig.add_subplot(233, projection='3d')
    ax_err_obs  = fig.add_subplot(234)
    ax_err_no_obs_attack = fig.add_subplot(235)
    ax_err_nominal = fig.add_subplot(236)

    for drone in drones:
        drone.marker, = ax_live_obs.plot([], [], [], marker='o', color=drone.color, markersize=6)
        drone.marker_actual, = ax_live_actual.plot([], [], [], marker='o', color=drone.color, markersize=6)
        drone.ref_marker, = ax_ref.plot([], [], [], marker='x', color=drone.color, markersize=6)
        drone.err_line, = ax_err_obs.plot([], [], color=drone.color)
        drone.err_no_line, = ax_err_no_obs_attack.plot([], [], color=drone.color)
    
    for drone in drones_nominal:
        drone.err_line, = ax_err_nominal.plot([], [], color=drone.color)

    timer_text_obs = ax_live_obs.text2D(0.05, 0.95, "Time: 0.0 s | Collisions: 0", transform=ax_live_obs.transAxes, fontsize=12)
    timer_text_actual = ax_live_actual.text2D(0.05, 0.95, "Time: 0.0 s | Collisions: 0", transform=ax_live_actual.transAxes, fontsize=12)

    traj_arr = np.array(trajectories)
    extra_bound = 5
    x_min, x_max = np.min(traj_arr[:,:,0])-extra_bound, np.max(traj_arr[:,:,0])+extra_bound
    y_min, y_max = np.min(traj_arr[:,:,1])-extra_bound, np.max(traj_arr[:,:,1])+extra_bound
    z_min, z_max = np.min(traj_arr[:,:,2])-extra_bound, np.max(traj_arr[:,:,2])+extra_bound
    for ax in [ax_live_obs, ax_ref, ax_live_actual]:
        ax.set_xlim([x_min, x_max]); ax.set_ylim([y_min, y_max]); ax.set_zlim([z_min, z_max])
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")

    ax_live_obs.set_title("Observed Drone Positions")
    ax_live_actual.set_title("Actual Drone Positions")
    ax_ref.set_title("Reference Positions")
    ax_err_obs.set_title("Desired - obs")
    ax_err_no_obs_attack.set_title("Desired - actual")
    ax_err_nominal.set_title("Obs - nominal")

    collisions_obs = 0
    collisions_actual = 0

    for k in range(N):
        for drone in drones:
            drone.step(drones)
        for drone in drones_nominal:
            drone.step(drones_nominal)

        collisions_obs += check_collisions(drones, use_obs=True, radius=1.0)
        collisions_actual += check_collisions(drones, use_obs=False, radius=1.0)

        if k % 20 == 0:
            sim_time = k * drones[0].dt
            for drone in drones:
                pos = np.array(drone.posHistory_obs)
                drone.marker.set_data([pos[-1,0]], [pos[-1,1]])
                drone.marker.set_3d_properties([pos[-1,2]])
                drone.marker_actual.set_data([drone.p[0,0]], [drone.p[1,0]])
                drone.marker_actual.set_3d_properties([drone.p[2,0]])
                ref = drone.nextPoint.flatten()
                drone.ref_marker.set_data([ref[0]], [ref[1]])
                drone.ref_marker.set_3d_properties([ref[2]])
                drone.err_line.set_data(np.arange(len(drone.errHistory_obs))*drone.dt, drone.errHistory_obs)
                drone.err_no_line.set_data(np.arange(len(drone.errHistory_pre_obs_attack))*drone.dt, drone.errHistory_pre_obs_attack)

            for i in range(len(drones)):
                drone = drones[i]
                drone_nominal = drones_nominal[i]
                drone_nominal.err_line.set_data(
                    np.arange(len(drone.errHistory_obs))*drone.dt,
                    np.array(drone.errHistory_obs) - np.array(drone_nominal.errHistory_obs)
                )

            timer_text_obs.set_text(f"Time: {sim_time:.1f} s | Collisions: {collisions_obs}")
            timer_text_actual.set_text(f"Time: {sim_time:.1f} s | Collisions: {collisions_actual}")
            ax_err_obs.relim(); ax_err_obs.autoscale_view()
            ax_err_no_obs_attack.relim(); ax_err_no_obs_attack.autoscale_view()
            ax_err_nominal.relim(); ax_err_nominal.autoscale_view()
            plt.pause(0.001)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    simulation()
