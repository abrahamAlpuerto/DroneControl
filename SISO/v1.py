import pybullet as p
import pybullet_data

import time
import numpy as np
import matplotlib.pyplot as plt

# pybullet setup
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
droneId = p.loadURDF("my_drone.urdf", [0, 0, 0.1])


# control params

Kp, Ki, Kd = 35, 20, 20
target_altitude = 2.0
integral = 0
last_e = 0
dt = 1/240.0 # default step time in pybullet

alt_history = []
time_history = []

print("Running Sim!")

for i in range(1000):
    pos, _ = p.getBasePositionAndOrientation(droneId) # x, y, z
    vel, _ = p.getBaseVelocity(droneId)
    cur_z = pos[2]


    #need error, the integral of error, and derivative or error
    error = target_altitude - cur_z
    integral += error * dt * (-0.1 < error < 0.1)
    derivative = (error - last_e) / dt

    print(f"step {i} | error {error} | integral {integral} | derivative {derivative}")
    # u(t) = Kp*error + Ki*integral + Kd*derivative + mg
    thrust = (Kp * error) + (Ki * integral) + (Kd * derivative) + 0.5 * 9.81

    #applying input to center of drone
    p.applyExternalForce(droneId, -1, [0, 0, thrust], [0, 0, 0], p.WORLD_FRAME)
    
    # log data
    alt_history.append(cur_z)
    time_history.append(i * dt)
    last_e = error


    p.stepSimulation()
    time.sleep(dt)

p.disconnect()



plt.figure(figsize=(10, 5))
plt.plot(time_history, alt_history, label='Actual Altitude', linewidth=2)

plt.axhline(y=target_altitude, color='r', linestyle='--', label=f'Target ({target_altitude}m)')

plt.axvline(x=1.5, color='g', linestyle=':', label='Settling Limit (1.5s)')
plt.title(f"SISO Altitude Control Performance ($K_p={Kp}, K_i={Ki}, K_d={Kd}$)")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.legend()
plt.grid(True)
plt.show()