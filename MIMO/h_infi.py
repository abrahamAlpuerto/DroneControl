import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.signal import StateSpace, lsim

def generate_step_response_comparison():
    print("Calculating Closed-Loop Dynamics...")
    # 1. System Dynamics
    m = 0.5; Iyy = 0.01; g = 9.81
    A = np.zeros((6, 6))
    A[0, 3] = 1.0; A[1, 4] = 1.0; A[2, 5] = 1.0; A[5, 1] = -g 
    B = np.zeros((6, 2))
    B[3, 0] = 1 / m; B[4, 1] = 1 / Iyy
    C_int = np.zeros((2, 6))
    C_int[0, 0] = 1.0; C_int[1, 2] = 1.0  
    A_aug = np.block([[A, np.zeros((6, 2))], [-C_int, np.zeros((2, 2))]])
    B_aug = np.block([[B], [np.zeros((2, 2))]])

    # 2. Controllers
    # Baseline (Manual Guess)
    Q_base = np.eye(8) * 10.0 
    R_base = np.eye(2)
    P_base = solve_continuous_are(A_aug, B_aug, Q_base, R_base)
    K_base = np.linalg.inv(R_base) @ B_aug.T @ P_base

    # ML-Optimized (Optuna)
    Q_opt = np.zeros((8, 8))
    Q_opt[0, 0] = 3594.6; Q_opt[1, 1] = 1866599.2; Q_opt[2, 2] = 2983428.6
    Q_opt[3, 3] = 51.7;   Q_opt[4, 4] = 2.45;      Q_opt[5, 5] = 21364.8
    Q_opt[6, 6] = 310.1;  Q_opt[7, 7] = 17.3
    R_opt = np.array([[0.428, 0], [0, 0.269]])
    P_opt = solve_continuous_are(A_aug, B_aug, Q_opt, R_opt)
    K_opt = np.linalg.inv(R_opt) @ B_aug.T @ P_opt

    # 3. Build the Reference Tracking System
    def create_tracking_sys(K_aug):
        A_cl = A_aug - B_aug @ K_aug
        Kp = K_aug[:, :6]
        
        # B_ref maps the target goals [z_target, x_target] into the dynamics
        B_ref = np.zeros((8, 2))
        B_ref[:6, 0] = (B @ Kp)[:, 0]  # Target Z affects proportional error
        B_ref[:6, 1] = (B @ Kp)[:, 2]  # Target X affects proportional error
        B_ref[6, 0] = 1.0              # Target Z drives the Integrator
        B_ref[7, 1] = 1.0              # Target X drives the Integrator
        
        # We only want to plot Z (index 0) and X (index 2)
        C_out = np.zeros((2, 8))
        C_out[0, 0] = 1.0
        C_out[1, 2] = 1.0
        D_out = np.zeros((2, 2))
        
        return StateSpace(A_cl, B_ref, C_out, D_out)

    sys_base = create_tracking_sys(K_base)
    sys_opt = create_tracking_sys(K_opt)

    # 4. Simulate a 10-Second Flight
    t = np.linspace(0, 10, 1000)
    u_targets = np.zeros((len(t), 2))
    u_targets[:, 0] = 2.0  # Command a 2.0m Altitude Step
    u_targets[:, 1] = 1.0  # Command a 1.0m X-Position Step

    print("Simulating Step Responses...")
    _, y_base, _ = lsim(sys_base, u_targets, t)
    _, y_opt, _ = lsim(sys_opt, u_targets, t)

    z_base = y_base[:, 0]; x_base = y_base[:, 1]
    z_opt = y_opt[:, 0];   x_opt = y_opt[:, 1]

    # 5. Calculate Metrics Function
    def calc_metrics(data_arr, target):
        peak = np.max(data_arr)
        os = max(0.0, ((peak - target) / target) * 100.0)
        
        lower_bound = target * 0.98; upper_bound = target * 1.02
        out_bounds = np.where((data_arr < lower_bound) | (data_arr > upper_bound))[0]
        
        if len(out_bounds) == 0: ts = 0.0
        elif out_bounds[-1] == len(data_arr) - 1: ts = float('inf')
        else: ts = t[out_bounds[-1] + 1]
        
        return os, ts

    # Print the Hard Numbers
    print("\n--- PERFORMANCE METRICS (2% Error Band) ---")
    os_b_z, ts_b_z = calc_metrics(z_base, 2.0); os_o_z, ts_o_z = calc_metrics(z_opt, 2.0)
    print("ALTITUDE (Z):")
    print(f"  Baseline  -> Overshoot: {os_b_z:6.2f}%, Settling Time: {ts_b_z:.2f}s")
    print(f"  Optimized -> Overshoot: {os_o_z:6.2f}%, Settling Time: {ts_o_z:.2f}s")

    os_b_x, ts_b_x = calc_metrics(x_base, 1.0); os_o_x, ts_o_x = calc_metrics(x_opt, 1.0)
    print("\nPOSITION (X):")
    print(f"  Baseline  -> Overshoot: {os_b_x:6.2f}%, Settling Time: {ts_b_x:.2f}s")
    print(f"  Optimized -> Overshoot: {os_o_x:6.2f}%, Settling Time: {ts_o_x:.2f}s")

    # 6. Plot the Graph
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Altitude Plot
    ax1.plot(t, z_base, label='Baseline Controller', color='red', linestyle='--')
    ax1.plot(t, z_opt, label='ML-Optimized Controller', color='blue', linewidth=2)
    ax1.axhline(y=2.0, color='black', linestyle=':', label='Target Altitude')
    ax1.set_title("Step Response: Altitude Tracking (Z)")
    ax1.set_ylabel("Altitude (m)")
    ax1.legend(); ax1.grid(True)

    # X-Position Plot
    ax2.plot(t, x_base, label='Baseline Controller', color='red', linestyle='--')
    ax2.plot(t, x_opt, label='ML-Optimized Controller', color='blue', linewidth=2)
    ax2.axhline(y=1.0, color='black', linestyle=':', label='Target Position')
    ax2.set_title("Step Response: Horizontal Tracking (X)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position (m)")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Calculate the closed-loop matrix
    A_cl_opt = A_aug - B_aug @ K_opt

    # Calculate the eigenvalues
    eigenvalues = np.linalg.eigvals(A_cl_opt)

    print("Closed-Loop Eigenvalues:")
    for i, eig in enumerate(eigenvalues):
        print(f"Pole {i+1}: {eig:.4f}")
        
    # Automated Stability Check
    is_stable = all(np.real(eig) < 0 for eig in eigenvalues)
    print(f"\nIs the system mathematically stable? {is_stable}")

if __name__ == "__main__":
    generate_step_response_comparison()