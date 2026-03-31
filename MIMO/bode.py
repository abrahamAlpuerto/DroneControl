import numpy as np
import matplotlib.pyplot as plt

def plot_pole_zero_map():
    eigenvalues = np.array([
        -393.0386 + 330.0619j,
        -393.0386 - 330.0619j,
        -19.9750 + 0.0000j,
        -9.1702 + 0.0000j,
        -0.2939 + 0.0000j,
        -2.5449 + 2.4343j,
        -2.5449 - 2.4343j,
        -0.0024 + 0.0000j
    ])

    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)

    plt.figure(figsize=(10, 6))
    
    plt.scatter(real_parts, imag_parts, marker='x', color='red', s=100, linewidths=2, label='Closed-Loop Poles')

    plt.axvline(x=0, color='black', linewidth=2)
    plt.axhline(y=0, color='black', linewidth=1)

    plt.axvspan(min(real_parts) - 20, 0, facecolor='green', alpha=0.1, label='Stable Region (LHP)')
    plt.axvspan(0, 50, facecolor='red', alpha=0.1, label='Unstable Region (RHP)')


    plt.title('Pole-Zero Map', fontsize=14)
    plt.xlabel('Real Axis', fontsize=12)
    plt.ylabel('Imaginary Axis', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_pole_zero_map()