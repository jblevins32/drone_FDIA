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

    # Take exactly 10 points (since we have 10 drones)
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

# --- Drone class ---
class Drone:
    def __init__(self, trajectory, color='b'):
        # Parameters
        self.m = 1.0
        self.g = 9.81
        self.J = np.diag([0.02, 0.02, 0.04])
        self.dt = 0.01  # larger dt for speed

        # Gains
        self.Kp = 2.0
        self.Kd = 3.0
        self.K_R = 500
        self.K_w = 20
        self.K_avoid = 2.0  # collision avoidance gain
        self.safety_radius = 1.0

        # State
        rand_x_pos = np.random.uniform(-5, 5)
        rand_y_pos = np.random.uniform(-5, 5)
        self.p = np.array([[rand_x_pos],[rand_y_pos],[0]])   # position
        self.v = np.zeros((3,1))
        self.R = np.eye(3)
        self.omega = np.zeros((3,1))

        # Trajectory
        self.trajectory = trajectory
        self.stage = 0
        self.nextPoint = self.trajectory[self.stage,:].reshape(3,1)

        # History
        self.posHistory = []

        # Visualization
        self.color = color
        self.line = None
        self.marker = None

    def step(self, others):
        dt, m, g, J = self.dt, self.m, self.g, self.J

        # --- Outer loop PID ---
        p_des = self.nextPoint

        # Avoidance term
        a_avoid = np.zeros((3,1))
        for other in others:
            if other is self: 
                continue
            diff = self.p - other.p
            dist = np.linalg.norm(diff)
            if dist < self.safety_radius and dist > 1e-3:
                a_avoid += self.K_avoid * diff / (dist**2)

        a_cmd = self.Kp*(p_des - self.p) - self.Kd*self.v + a_avoid
        a_cmd[2] += g  # gravity compensation

        # --- Desired orientation ---
        f_des = m*a_cmd
        T = np.linalg.norm(f_des)
        if T < 1e-6: T = 1e-6
        b3_des = f_des / T
        psi_des = 0.0
        b1_psi = np.array([[np.cos(psi_des)], [np.sin(psi_des)], [0]])
        b2_des = np.cross(b3_des.flatten(), b1_psi.flatten())
        b2_des /= np.linalg.norm(b2_des)
        b1_des = np.cross(b2_des, b3_des.flatten())
        R_des = np.column_stack((b1_des, b2_des, b3_des.flatten()))

        # --- Inner loop attitude ---
        e_R_mat = 0.5*(R_des.T @ self.R - self.R.T @ R_des)
        e_R = np.array([[e_R_mat[2,1]], [e_R_mat[0,2]], [e_R_mat[1,0]]])
        tau = -self.K_R*e_R - self.K_w*self.omega + \
              np.cross(self.omega.flatten(), (J @ self.omega).flatten()).reshape(3,1)

        # Dynamics
        omega_dot = np.linalg.inv(J) @ (tau - np.cross(self.omega.flatten(), (J @ self.omega).flatten()).reshape(3,1))
        self.omega += omega_dot*dt
        self.R = self.R @ expm(hat((self.omega*dt).flatten()))
        acc = np.array([[0],[0],[-g]]) + (T/m)*self.R[:,2].reshape(3,1)
        self.v += acc*dt
        self.p += self.v*dt

        # History
        self.posHistory.append(self.p.flatten())

        # Waypoint switching (move to next digit after some time)
        # Each digit lasts ~5s
        if len(self.posHistory)*dt > (self.stage+1)*5 and self.stage < len(self.trajectory)-1:
            self.stage += 1
            self.nextPoint = self.trajectory[self.stage,:].reshape(3,1)

# --- Simulation ---
def simulation():
    Tsim = 30
    N = round(Tsim/0.01)

    # --- Generate digit sequences ---
    trajectories = generate_digit_sequences()

    # --- Create 10 drones ---
    colors = plt.cm.tab10(np.linspace(0,1,10))
    drones = [Drone(trajectories[i], color=colors[i]) for i in range(10)]

    # --- Setup plotting ---
    plt.ion()
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    for drone in drones:
        drone.line, = ax.plot([], [], [], color=drone.color, linewidth=1.0)
        drone.marker, = ax.plot([], [], [], marker='o', color=drone.color, markersize=6)

    timer_text = ax.text2D(0.05, 0.95, "Time: 0.0 s", transform=ax.transAxes, fontsize=12)

    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title("10 Drones Forming Numbers 1-5")
    ax.set_box_aspect([1,1,1])

    # --- Simulation loop ---
    for k in range(N):
        for drone in drones:
            drone.step(drones)

        if k % 20 == 0:  # update plot
            sim_time = k * drones[0].dt
            for drone in drones:
                pos = np.array(drone.posHistory)
                drone.line.set_data(pos[:,0], pos[:,1])
                drone.line.set_3d_properties(pos[:,2])
                drone.marker.set_data([pos[-1,0]], [pos[-1,1]])
                drone.marker.set_3d_properties([pos[-1,2]])
            timer_text.set_text(f"Time: {sim_time:.1f} s")
            plt.pause(0.001)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    simulation()
