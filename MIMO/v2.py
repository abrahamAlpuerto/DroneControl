import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are


m = 0.5 
Iyy = 0.01 
g = 9.81


A = np.zeros((6, 6))
A[0, 3] = 1.0; A[1, 4] = 1.0; A[2, 5] = 1.0
A[5, 1] = -g 

B = np.zeros((6, 2))
B[3, 0] = 1 / m
B[4, 1] = 1 / Iyy


C = np.zeros((2, 6))
C[0, 0] = 1.0 
C[1, 2] = 1.0  


n_states = A.shape[0]   
n_outputs = C.shape[0] 

A_aug = np.block([
    [A, np.zeros((n_states, n_outputs))],
    [-C, np.zeros((n_outputs, n_outputs))]
])


B_aug = np.block([
    [B],
    [np.zeros((n_outputs, B.shape[1]))]
])


Q_aug = np.zeros((8, 8))



Q_aug[0, 0] = 1000.0  
Q_aug[3, 3] = 500.0 

Q_aug[1, 1] = 2500000.0  
Q_aug[4, 4] = 1.0  

Q_aug[2, 2] = 2000000.0
Q_aug[5, 5] = 1750000.0 


Q_aug[6, 6] = 100.0    
Q_aug[7, 7] = 5000.0   

R = np.eye(2) 


P_aug = solve_continuous_are(A_aug, B_aug, Q_aug, R)
K_aug = np.linalg.inv(R) @ B_aug.T @ P_aug


Kp = K_aug[:, :6]
Ki = -K_aug[:, 6:] 

print("Calculated LQI Proportional Gain Kp:\n", Kp)
print("Calculated LQI Integral Gain Ki:\n", Ki)


physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -g)
p.loadURDF("plane.urdf")


start_pos = [1, 0, 1]
droneId = p.loadURDF("my_drone.urdf", start_pos)

dt = 1/240.0
p.setTimeStep(dt)


target_z = 2.0
target_x = 0.0
target_state = np.array([target_z, 0.0, target_x, 0.0, 0.0, 0.0])

alt_history = []
x_history = []  
time_history = []

print("Running MIMO LQR Sim!")


error_int_z = 0.0
error_int_x = 0.0


print("Running MIMO LQI Sim!")

for i in range(3000):

    pos, ori = p.getBasePositionAndOrientation(droneId)
    vel, ang_vel = p.getBaseVelocity(droneId)
    euler = p.getEulerFromQuaternion(ori)
    

    z = pos[2]
    theta = euler[1] 
    x_pos = pos[0]
    z_dot = vel[2]
    theta_dot = ang_vel[1]
    x_dot = vel[0]
    
    current_state = np.array([z, theta, x_pos, z_dot, theta_dot, x_dot])
    

    err_z = target_z - z
    err_x = target_x - x_pos


    if z > 0.2 and abs(target_z - z) < 0.2:
        error_int_z += err_z * dt
        

    if z > 0.5 and abs(target_x - x_pos) < 0.1:
        error_int_x += err_x * dt


    error_int_z = np.clip(error_int_z, -2.0, 2.0)
    error_int_x = np.clip(error_int_x, -5.0, 5.0)
    

    integral_states = np.array([error_int_z, error_int_x])
    

    error_p = current_state - target_state
    u = -np.dot(Kp, error_p) + np.dot(Ki, integral_states)
    

    U1 = u[0]  # Vertical thrust effort
    U3 = u[1]  # Pitch torque effort
    thrust = U1 + (m * g)
    pitch_torque = U3
    

    F_x = -thrust * np.sin(theta) 
    F_z = thrust * np.cos(theta)
    
    p.applyExternalForce(droneId, -1, [F_x, 0, F_z], [0, 0, 0], p.WORLD_FRAME)
    p.applyExternalTorque(droneId, -1, [0, pitch_torque, 0], p.WORLD_FRAME)
    

    alt_history.append(z)
    time_history.append(i * dt)
    x_history.append(x_pos)

    p.stepSimulation()
    time.sleep(dt)

p.disconnect()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))


ax1.plot(time_history, alt_history, label='Actual Altitude (z)', linewidth=2)
ax1.axhline(y=target_z, color='r', linestyle='--', label=f'Target Z ({target_z}m)')
ax1.set_title("MIMO LQR Control Performance")
ax1.set_ylabel("Altitude (m)")
ax1.legend()
ax1.grid(True)


ax2.plot(time_history, x_history, label='Actual Position (x)', linewidth=2, color='green')
ax2.axhline(y=target_x, color='r', linestyle='--', label=f'Target X ({target_x}m)')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("X Position (m)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()