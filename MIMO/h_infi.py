import numpy as np
import control
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

m, Iyy, g = 0.5, 0.01, 9.81 
#[z, theta, x, z_dot, theta_dot, x_dot]
A = np.zeros((6, 6))
A[0, 3] = 1.0; A[1, 4] = 1.0; A[2, 5] = 1.0
A[5, 1] = -g 

B = np.zeros((6, 2))
B[3, 0] = 1 / m; B[4, 1] = 1 / Iyy

# C_int extracts z and x for error integration
C_int = np.zeros((2, 6))
C_int[0, 0] = 1.0; C_int[1, 2] = 1.0

#Aug Matrices (8x8 and 8x2)
A_aug = np.block([[A, np.zeros((6, 2))], [-C_int, np.zeros((2, 2))]])
B_aug = np.block([[B], [np.zeros((2, 2))]])

#Optimal Weights from Optuna Optimization
# Mapping to: [z, theta, x, z_dot, theta_dot, x_dot, int_z, int_x] 
Q_diag = [
    5419.373331099708,   # z
    172629.40003522372,  # theta
    3622551.9403309003,  # x
    50.14212329709968,   # z_dot
    1.294699297199231,   # theta_dot
    462893.84742157307,  # x_dot
    3.2544350393384507,  # integral_z
    4863.273320495843    # integral_x
]
Q_aug = np.diag(Q_diag)
R = np.diag([0.23676311101793757, 5.842725136204356]) # [thrust, pitch]

#Controller Synthesis H2
P_ric = solve_continuous_are(A_aug, B_aug, Q_aug, R)
K_aug = np.linalg.inv(R) @ B_aug.T @ P_ric

#Stability Verification
A_cl = A_aug - B_aug @ K_aug
poles = np.linalg.eigvals(A_cl)

print("Exact Closed-Loop Poles:")
for i, pole in enumerate(poles):
    print(f"Pole {i+1}: {pole.real:.4e} + {pole.imag:.4e}j")

#pole zeros


plt.figure(figsize=(8, 6))
plt.scatter(poles.real, poles.imag, marker='x', color='red', label='CL Poles')
plt.axvline(0, color='black', linestyle='--')
plt.title("Stability Verification: Augmented Pole-Zero Map")
plt.xlabel("Real Axis"); plt.ylabel("Imaginary Axis")
plt.grid(True); plt.legend()

plt.xlim(right=5) 
plt.savefig("stability_verification.png")

# sigma plot
f = np.logspace(-2, 3, 500)


sys_P = control.ss(A_aug, B_aug, np.eye(8), 0)
sys_K = control.ss(np.zeros((2,2)), np.zeros((2,8)), np.zeros((2,2)), K_aug)
L = sys_P * sys_K
S = control.feedback(np.eye(2), L[:2, :2]) # Primary input-output loop

sv_resp = control.singular_values_response(S, omega=f)

mag = np.squeeze(sv_resp.magnitude)  

plt.figure(figsize=(8, 6))

plt.semilogx(f, 20*np.log10(mag[0, :]), label='Max SV ($\sigma_{max}$)')
plt.semilogx(f, 20*np.log10(mag[1, :]), label='Min SV ($\sigma_{min}$)')

plt.axhline(0, color='black', linestyle='--')
plt.title("Robustness Evaluation: Sensitivity Sigma Plot")
plt.xlabel("Frequency (rad/s)"); plt.ylabel("Magnitude (dB)")
plt.grid(True, which="both"); plt.legend()
plt.savefig("robustness_bounds.png")