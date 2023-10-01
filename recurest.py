import numpy as np
import matplotlib.pyplot as plt

K = 3

# Define the functions f and g. Example:
def f(z):
    return (K - 1) * z / (K + 1)

def g(z):
    return z / (K + 1 - (K - 1) * z)

def compute_depth(z, f, g, max_depth):
    if np.abs(z) < 0.5:
        return 0
    if max_depth == 0:
        return max_depth
    
    # Compute the next depth level for both f and g
    depth_f = 1 + compute_depth(f(z), f, g, max_depth - 1)
    depth_g = 1 + compute_depth(g(z), f, g, max_depth - 1)

    return max(depth_f, depth_g)

def plot_depth_field(x_lim=(-2, 2), y_lim=(-2, 2), resolution=100, max_depth=10):
    x = np.linspace(x_lim[0], x_lim[1], resolution)
    y = np.linspace(y_lim[0], y_lim[1], resolution)
    xs, ys = np.meshgrid(x, y)
    zs = xs + 1j * ys

    depth_field = np.zeros_like(xs, dtype=int)

    for i in range(resolution):
        for j in range(resolution):
            depth_field[i, j] = compute_depth(zs[i, j], f, g, max_depth)

    im = plt.imshow(depth_field, extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]), origin='lower', cmap='viridis', interpolation='nearest')
    
    # Update the colorbar to have integer ticks
    cbar = plt.colorbar(im, label='Depth', ticks=np.arange(max_depth + 1))
    cbar.set_ticklabels(np.arange(max_depth + 1))  # Set tick labels to integer values
    
    plt.axhline(0, color='white', linewidth=0.5)
    plt.axvline(0, color='white', linewidth=0.5)
    plt.title('Minimum Composition Depth Field')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.show()

plot_depth_field(x_lim=(-5, 5), y_lim=(-5, 5), resolution=800, max_depth=30)
