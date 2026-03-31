import optuna
import numpy as np
import control
import pybullet as p
import time
import pybullet_data
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are


import numpy as np
import control

def build_generalized_plant(weight_z, weight_theta, weight_x, weight_thrust, weight_pitch):

    m = 0.5 
    Iyy = 0.01 
    g = 9.81

    # State: x = [z, theta, x, z_dot, theta_dot, x_dot]
    A = np.zeros((6, 6))
    A[0, 3] = 1.0; A[1, 4] = 1.0; A[2, 5] = 1.0
    A[5, 1] = -g 

    B2 = np.zeros((6, 2))
    B2[3, 0] = 1 / m
    B2[4, 1] = 1 / Iyy


    # w = [wind_z, wind_theta, wind_x]
    n_states = 6
    n_disturbances = 3
    B1 = np.zeros((n_states, n_disturbances))
    B1[3, 0] = 0.1  # Wind effect on Z acceleration
    B1[4, 1] = 0.1  # Wind effect on Pitch acceleration
    B1[5, 2] = 0.1  # Wind effect on X acceleration


    # We want to minimize: z, theta, x, and control efforts (u1, u2)
    # Output vector z_perf has 5 rows: [weighted_z, weighted_theta, weighted_x, weighted_u1, weighted_u2]
    n_perf = 5
    n_controls = 2
    
    C1 = np.zeros((n_perf, n_states))
    C1[0, 0] = weight_z      # Penalize altitude error
    C1[1, 1] = weight_theta  # Penalize pitch angle
    C1[2, 2] = weight_x      # Penalize horizontal position error

    D11 = np.zeros((n_perf, n_disturbances)) 

    D12 = np.zeros((n_perf, n_controls))
    D12[3, 0] = weight_thrust # Penalize Thrust
    D12[4, 1] = weight_pitch  # Penalize Pitch Torque

  
    n_meas = 6
    C2 = np.eye(n_meas)
    

    # We add a tiny assumed sensor noise to make the math robust
    D21 = np.eye(n_meas, n_disturbances) * 0.001 
    
    D22 = np.zeros((n_meas, n_controls))


    # P = [ A   | B1   B2  ]
    #     [ C1  | D11  D12 ]
    #     [ C2  | D21  D22 ]
    
    B_P = np.block([[B1, B2]])
    C_P = np.block([[C1], 
                    [C2]])
    D_P = np.block([[D11, D12], 
                    [D21, D22]])

    P = control.ss(A, B_P, C_P, D_P)
    
    return P



def run_pybullet_sim(K_aug, show_gui=False, return_history=False):

    mode = p.GUI if show_gui else p.DIRECT
    physicsClient = p.connect(mode)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    start_pos = [0, 0, 1]
    droneId = p.loadURDF("my_drone.urdf", start_pos)
    
    dt = 1/240.0
    p.setTimeStep(dt)
    
 
    m = 0.5 
    g = 9.81
    
    
    target_z =  1
    target_x = 0

    target_state = np.array([target_z, 0.0, target_x, 0.0, 0.0, 0.0])
    

    Kp = K_aug[:, :6]
    Ki = -K_aug[:, 6:] 
    

    error_int_z = 0.0
    error_int_x = 0.0
    total_cost = 0.0  


    alt_history = []
    x_history = []
    time_history = []
    
    total_cost = 0.0


    
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
        
        U1 = u[0]
        U3 = u[1]
        
        thrust = U1 + (m * g)
        pitch_torque = U3
        
        F_x = -thrust * np.sin(theta) 
        F_z = thrust * np.cos(theta)
        
        p.applyExternalForce(droneId, -1, [F_x, 0, F_z], [0, 0, 0], p.WORLD_FRAME)
        p.applyExternalTorque(droneId, -1, [0, pitch_torque, 0], p.WORLD_FRAME)
        

        step_cost = (err_z**2) + (err_x**2)
        

        if z < 0.1 or abs(theta) > 1.5:
            step_cost += 10000.0 
            
        total_cost += step_cost
        
        if return_history:
            alt_history.append(z)
            x_history.append(x_pos)
            time_history.append(i * dt)
            
        p.stepSimulation()
        if show_gui:
            time.sleep(dt)
            
    p.disconnect()
    

    if return_history:
        return time_history, alt_history, x_history
    else:
        return total_cost


def objective(trial):

    weight_z = trial.suggest_float("weight_z", 100.0, 10000.0, log=True)
    weight_z_dot = trial.suggest_float("weight_z_dot", 10.0, 1000.0, log=True)
    
    weight_theta = trial.suggest_float("weight_theta", 10000.0, 5000000.0, log=True)
    weight_theta_dot = trial.suggest_float("weight_theta_dot", 1.0, 500.0, log=True)
    
    weight_x = trial.suggest_float("weight_x", 10000.0, 5000000.0, log=True)
    weight_x_dot = trial.suggest_float("weight_x_dot", 1000.0, 1000000.0, log=True)
    
    weight_int_z = trial.suggest_float("weight_int_z", 1.0, 500.0, log=True)
    weight_int_x = trial.suggest_float("weight_int_x", 10.0, 10000.0, log=True)
    
    weight_thrust = trial.suggest_float("weight_thrust", 0.1, 10.0)
    weight_pitch = trial.suggest_float("weight_pitch", 0.1, 10.0)

    m = 0.5 
    Iyy = 0.01 
    g = 9.81

    A = np.zeros((6, 6))
    A[0, 3] = 1.0; A[1, 4] = 1.0; A[2, 5] = 1.0
    A[5, 1] = -g 

    B = np.zeros((6, 2))
    B[3, 0] = 1 / m
    B[4, 1] = 1 / Iyy

    C_int = np.zeros((2, 6))
    C_int[0, 0] = 1.0  
    C_int[1, 2] = 1.0  

    A_aug = np.block([[A, np.zeros((6, 2))], [-C_int, np.zeros((2, 2))]])
    B_aug = np.block([[B], [np.zeros((2, 2))]])

    Q_aug = np.zeros((8, 8))
    Q_aug[0, 0] = weight_z
    Q_aug[1, 1] = weight_theta
    Q_aug[2, 2] = weight_x
    Q_aug[3, 3] = weight_z_dot
    Q_aug[4, 4] = weight_theta_dot
    Q_aug[5, 5] = weight_x_dot
    Q_aug[6, 6] = weight_int_z
    Q_aug[7, 7] = weight_int_x

    R = np.array([[weight_thrust, 0], 
                  [0, weight_pitch]])

    try:
        P_aug = solve_continuous_are(A_aug, B_aug, Q_aug, R)
        K_aug = np.linalg.inv(R) @ B_aug.T @ P_aug
        
    except Exception as e:

        return 9999999.0


    sim_error = run_pybullet_sim(K_aug, show_gui=False)
    
    return sim_error



def calculate_metrics(time_hist, data_hist, target_val, tolerance=0.02):

    data = np.array(data_hist)
    time_array = np.array(time_hist)
    

    if target_val == 0.0:
        peak_dev = np.max(np.abs(data))
        os_display = f"{peak_dev:.4f} m (Absolute Max Deviation)"
        band_limit = 0.02 
        lower_bound = -band_limit
        upper_bound = band_limit
    else:
        peak_val = np.max(data)
        overshoot = ((peak_val - target_val) / target_val) * 100.0
        overshoot = max(0.0, overshoot) 
        os_display = f"{overshoot:.2f}%"
        

        lower_bound = target_val * (1 - tolerance)
        upper_bound = target_val * (1 + tolerance)
        

    outside_bounds = np.where((data < lower_bound) | (data > upper_bound))[0]
    
    if len(outside_bounds) == 0:
        settling_time = 0.0
    else:
        last_outside_idx = outside_bounds[-1]
        

        if last_outside_idx == len(data) - 1:
            settling_time = float('inf') 
        else:

            settling_time = time_array[last_outside_idx + 1]
            
    ts_display = f"{settling_time:.3f} s" if settling_time != float('inf') else "Did not settle"
    
    return os_display, ts_display




if __name__ == "__main__":
    rng = np.random.default_rng()

    

    study = optuna.create_study(direction="minimize")

    study.optimize(objective, n_trials=150)
    
    print("\nOptimization Finished!")
    print("Best Simulator Score:", study.best_value)
    print("Optimal Controller Weights:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\n--- Generating Final Optimal Controller ---")


    best = study.best_params

    m = 0.5; Iyy = 0.01; g = 9.81
    A = np.zeros((6, 6))
    A[0, 3] = 1.0; A[1, 4] = 1.0; A[2, 5] = 1.0; A[5, 1] = -g 
    B = np.zeros((6, 2))
    B[3, 0] = 1 / m; B[4, 1] = 1 / Iyy
    C_int = np.zeros((2, 6))
    C_int[0, 0] = 1.0; C_int[1, 2] = 1.0  

    A_aug = np.block([[A, np.zeros((6, 2))], [-C_int, np.zeros((2, 2))]])
    B_aug = np.block([[B], [np.zeros((2, 2))]])

    Q_opt = np.zeros((8, 8))
    Q_opt[0, 0] = best['weight_z']
    Q_opt[1, 1] = best['weight_theta']
    Q_opt[2, 2] = best['weight_x']
    Q_opt[3, 3] = best['weight_z_dot']
    Q_opt[4, 4] = best['weight_theta_dot']
    Q_opt[5, 5] = best['weight_x_dot']
    Q_opt[6, 6] = best['weight_int_z']
    Q_opt[7, 7] = best['weight_int_x']

    R_opt = np.array([[best['weight_thrust'], 0], 
                    [0, best['weight_pitch']]])


    P_opt = solve_continuous_are(A_aug, B_aug, Q_opt, R_opt)
    K_opt = np.linalg.inv(R_opt) @ B_aug.T @ P_opt

    print("\nOptimal Kp (Proportional Gains):\n", K_opt[:, :6])
    print("\nOptimal Ki (Integral Gains):\n", -K_opt[:, 6:])


    print("\nLaunching PyBullet visualization and generating plots...")


    time_hist, alt_hist, x_hist = run_pybullet_sim(K_opt, show_gui=True, return_history=True)


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    target_z = 2.0
    target_x = 0.0

    ax1.plot(time_hist, alt_hist, label='Optimal Altitude (z)', color='blue', linewidth=2)
    ax1.axhline(y=target_z, color='red', linestyle='--', label=f'Target Z ({target_z}m)')
    ax1.set_title("Optimized LQI Controller: Transient Response")
    ax1.set_ylabel("Altitude (m)")
    ax1.legend()
    ax1.grid(True)


    ax2.plot(time_hist, x_hist, label='Optimal Position (x)', color='green', linewidth=2)
    ax2.axhline(y=target_x, color='red', linestyle='--', label=f'Target X ({target_x}m)')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("X Position (m)")
    ax2.legend()
    ax2.grid(True)


    plt.tight_layout()
    plt.show()


    print("\n--- Final H2 Controller Performance Metrics ---")


    z_os, z_ts = calculate_metrics(time_hist, alt_hist, target_z)
    print(f"Altitude (Z) Target: {target_z}m")
    print(f"  Overshoot:     {z_os}")
    print(f"  Settling Time: {z_ts}")

    x_os, x_ts = calculate_metrics(time_hist, x_hist, target_x)
    print(f"\nPosition (X) Target: {target_x}m")
    print(f"  Max Deviation: {x_os}")
    print(f"  Settling Time: {x_ts}")
    print("-----------------------------------------------")