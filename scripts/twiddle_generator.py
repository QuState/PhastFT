from matplotlib import cm


def gen_twiddles(dist: int) -> list[complex]:
    theta = -np.pi / dist
    g = complex(np.cos(theta), np.sin(theta))

    w = complex(1.0, 0.0)
    print(w)

    for k in range(1, dist):
        w *= g
        print(w)


import matplotlib.pyplot as plt
import numpy as np


def cycle_colors(num_colors, colormap="viridis"):
    """
    Generate a list of colors by cycling over a specified colormap.

    Parameters:
    - num_colors (int): Number of colors to generate.
    - colormap (str): Name of the Matplotlib colormap to use.

    Returns:
    - colors (list): List of color values in hexadecimal format.
    """
    # Get the specified colormap
    cmap = cm.get_cmap(colormap)

    # Generate equally spaced values from 0 to 1
    values = np.linspace(0, 1, num_colors)

    # Map the values to colors using the colormap
    colors = [cmap(value) for value in values]

    # Convert colors to hexadecimal format
    hex_colors = [
        f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"
        for r, g, b, _ in colors
    ]

    return hex_colors


def plot_roots_of_unity(dist: int):
    # Calculate the nth roots of unity
    theta = -np.pi / dist
    roots = []
    g = complex(np.cos(theta), np.sin(theta))
    w = complex(1.0, 0.0)
    roots = [w]

    for k in range(1, dist):
        w *= g
        roots.append(w)

    roots = np.asarray(roots)
    for r in roots:
        print(f"{np.round(r.real, 2)} {np.round(r.imag, 2)}")

    temp = cycle_colors(dist // 2)
    temp.reverse()
    all_colors = cycle_colors(dist // 2) + temp
    all_colors[0] = "r"

    # Plot the roots
    plt.figure(figsize=(6, 6))
    plt.scatter(roots.real, roots.imag, color=all_colors, marker="o")
    plt.title(f"Roots of Unity (n={n})")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.axis("equal")

    # Limit the axes to -1.0 to 1.0
    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)

    plt.show()


# # Specify the number of roots (n)
# n = 8  # You can change this to any positive integer
#
# # Plot the roots of unity for the specified value of n
# plot_roots_of_unity(n)


def main():
    # gen_twiddles(4)
    print("===================\n")
    gen_twiddles(16)


if __name__ == "__main__":
    main()

# (1+0j)
# (0.9238795325112867-0.3826834323650898j)
# (0.7071067811865475-0.7071067811865476j)
# (0.38268343236508967-0.9238795325112867j)
# (-1.2420623018332135e-16-0.9999999999999999j)
# (-0.38268343236508984-0.9238795325112866j)
# (-0.7071067811865476-0.7071067811865474j)
# (-0.9238795325112867-0.38268343236508956j)
