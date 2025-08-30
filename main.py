import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# --- Helper: skew-symmetric "hat" operator ---
def hat(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

# --- Generate smiley face trajectory components ---
def smiley_parts(z=10, center=(0,0), radius=10, n_points=200):
    cx, cy = center

    # Left eye
    eye_r = radius * 0.1
    eye_offset_x = -radius*0.3
    eye_offset_y = radius*0.3
    theta_eye = np.linspace(0, 2*np.pi, n_points//10)
    left_eye = np.vstack((cx + eye_offset_x + eye_r*np.cos(theta_eye),
                          cy + eye_offset_y + eye_r*np.sin(theta_eye),
                          np.ones_like(theta_eye)*z)).T

    # Right eye
    right_eye = np.vstack((cx + radius*0.3 + eye_r*np.cos(theta_eye),
                           cy + radius*0.3 + eye_r*np.sin(theta_eye),
                           np.ones_like(theta_eye)*z)).T

    # Smile arc
    offset = 3
    theta_smile = np.linspace(np.pi/6, 5*np.pi/6, n_points//2)
    smile_r = radius*0.5
    smile = np.vstack((cx + smile_r*np.cos(theta_smile),
                       cy - radius*0.2 + smile_r*np.sin(theta_smile) - offset,
                       np.ones_like(theta_smile)*z)).T

    return left_eye, right_eye, smile


# --- Drone class ---
class Drone:
    def __init__(self, trajectory, color='b'):
        # Parameters
        self.m = 1.0
        self.g = 9.81
        self.J = np.diag([0.02, 0.02, 0.04])
        self.dt = 0.001

        # Gains
        self.Kp = 4.0
        self.Kd = 5.0
        self.K_R = 1000
        self.K_w = 20

        # State
        rand_x_pos = np.random.uniform(0, 10)
        rand_y_pos = np.random.uniform(0, 10)
        self.p = np.array([[rand_x_pos],[rand_y_pos],[0]])   # position
        self.v = np.zeros((3,1))
        self.R = np.eye(3)
        self.omega = np.zeros((3,1))

        # Trajectory
        self.trajectory = trajectory
        self.trajIndex = 0
        self.nextPoint = self.trajectory[1,:].reshape(3,1)

        # History
        self.posHistory = []

        # Visualization
        self.color = color
        self.line = None
        self.ref_line = None

    def step(self):
        dt, m, g, J = self.dt, self.m, self.g, self.J

        # --- Outer loop PID ---
        p_des = self.nextPoint
        a_cmd = self.Kp*(p_des - self.p) - self.Kd*self.v
        a_cmd[2] += g

        # --- Desired orientation ---
        f_des = m*a_cmd
        T = np.linalg.norm(f_des)
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

        # Waypoint switching
        if np.linalg.norm(self.p - self.nextPoint) < 0.5:
            if self.trajIndex < self.trajectory.shape[0]-2:
                self.trajIndex += 1
                self.nextPoint = self.trajectory[self.trajIndex+1,:].reshape(3,1)

def attack():
    pass

def smsf():
    pass

def simulation():
    dt = 0.001
    Tsim = 30
    N = round(Tsim/dt)

    # --- Trajectories ---
    left_eye, right_eye, smile = smiley_parts(z=10, center=(5,5), radius=10)

    # --- Create drones ---
    drones = [
        Drone(smile, color='g'),
        Drone(left_eye, color='r'),
        Drone(right_eye, color='m')
    ]

    # --- Setup plotting ---
    plt.ion()
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    for drone in drones:
        # Path trace
        drone.line, = ax.plot([], [], [], drone.color, linewidth=1.0, label=f"Drone {drone.color}")
        # Reference path
        drone.ref_line, = ax.plot(drone.trajectory[:,0], drone.trajectory[:,1], drone.trajectory[:,2],
                                  '--', color=drone.color, linewidth=0.8)
        # Drone marker (point)
        drone.marker, = ax.plot([], [], [], marker='o', color=drone.color, markersize=8)

    # Timer text overlay (upper left corner of 3D plot)
    timer_text = ax.text2D(0.05, 0.95, "Time: 0.0 s", transform=ax.transAxes, fontsize=12)

    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title("3 Drones Drawing Smiley Face")
    # ax.legend()
    ax.set_box_aspect([1,1,1])

    # --- Simulation loop ---
    for k in range(N):
        for drone in drones:
            drone.step()

        # Update plots every 200 steps
        if k % 200 == 0:
            sim_time = k * dt
            for drone in drones:
                pos = np.array(drone.posHistory)
                # Update trail
                drone.line.set_data(pos[:,0], pos[:,1])
                drone.line.set_3d_properties(pos[:,2])
                # Update current drone marker
                drone.marker.set_data([pos[-1,0]], [pos[-1,1]])
                drone.marker.set_3d_properties([pos[-1,2]])

            # Update timer
            timer_text.set_text(f"Time: {sim_time:.1f} s")

            plt.pause(0.001)

    plt.ioff()
    plt.show()



if __name__ == "__main__":
    simulation()
