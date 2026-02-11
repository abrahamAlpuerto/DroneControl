import control as ct
import matplotlib.pyplot as plt
import numpy as np

Kp, Ki, Kd = 35, 20, 20
s = ct.TransferFunction.s
G = 2 / s**2

#Transient/Climb (PD Only)
L_pd = (Kd * s + Kp) * G

#Steady-State/Hover (Full PID)
L_pid = ((Kd * s**2 + Kp * s + Ki) / s) * G

#BODE 
plt.figure(figsize=(10, 8))
ct.bode_plot(L_pd, label="Mode A: Climb (PD)", dB=True)
ct.bode_plot(L_pid, label="Mode B: Hover (PID)", dB=True)
plt.legend()
plt.suptitle("Bode Comparison: Switching Control Modes")

#Nyquist Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ct.nyquist_plot(L_pd, ax=ax1)
ax1.set_title("Mode A: Climb (Type 2 PD)")
ax1.grid(True)

ct.nyquist_plot(L_pid, ax=ax2)
ax2.set_title("Mode B: Hover (Type 3 PID)")
ax2.grid(True)

plt.tight_layout()
plt.show()

#Terminal Stability
def print_margins(name, L):
    gm, pm, wg, wp = ct.margin(L)
    print(f"\n--- {name} Stability ---")
    print(f"Gain Margin: {20*np.log10(gm) if gm > 0 else 'Inf'} dB")
    print(f"Phase Margin: {pm:.2f} degrees")

print_margins("MODE A (CLIMB)", L_pd)
print_margins("MODE B (HOVER)", L_pid)