import numpy as np
import matplotlib.pyplot as plt

def plot_complex_region(K, x_lim=(-2, 2), y_lim=(-2, 2)):
    x = np.linspace(x_lim[0], x_lim[1], 400)
    y = np.linspace(y_lim[0], y_lim[1], 400)

    xs, ys = np.meshgrid(x, y)
    zs = xs + 1j * ys

    cond_1 = np.abs(zs / (K + 1 - (K - 1) * zs)) < 1/2

    region = cond_1

    plt.figure(figsize=(8,8))
    plt.imshow(region, extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]), origin='lower', cmap='Greys', alpha=0.3)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    plt.title(f"Region where convergence may be possible, K={K}")
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.show()

# Example usage:
plot_complex_region(K=3, x_lim=(-5, 5), y_lim=(-5,5))
