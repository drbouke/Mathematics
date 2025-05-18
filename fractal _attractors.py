import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
import os

# --------------------------
# 1. Define nonlinear functions
# --------------------------

def f1(x, y):
    return np.sin(x * y), np.cos(y) / 2

def f2(x, y):
    return x / 2 + np.sin(y), y**2 / 3 - 0.1

def f3(x, y):
    return np.tanh(x + y), np.sin(x - y)

def f4(x, y):
    return np.sin(x) * np.cos(y), np.tanh(x - y)

def f5(x, y):
    return np.sin(x**2 - y), np.cos(x * y)

def f6(x, y):
    return np.tanh(x) + y / 2, np.cos(y) - x / 3

def f7(x, y):
    return np.sin(x + y**2), np.tanh(y - x)

def f8(x, y):
    return np.sin(x*y) - np.cos(y), np.sin(y**2 + x)

def f9(x, y):
    return np.tanh(x*y) + 0.1, np.sin(y) / 2

def f10(x, y):
    return np.cos(x**2 + y**2), np.sin(x - y)

def f11(x, y):
    return np.sin(3 * x) * np.cos(2 * y), np.tanh(x + y)

def f12(x, y):
    return x * np.cos(y), y * np.sin(x)

# --------------------------
# 2. Core simulation
# --------------------------

def simulate_rnifs(functions, probabilities, num_points=100000, burn_in=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x, y = 0.1, 0.1
    points = []

    for _ in range(num_points + burn_in):
        func = np.random.choice(functions, p=probabilities)
        x, y = func(x, y)
        if _ >= burn_in:
            points.append((x, y))

    return np.array(points)

# --------------------------
# 3. Visualization
# --------------------------

def plot_fractal(points, filename='fractal.png', title=None):
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=0.1, color='black')
    if title:
        plt.title(title)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_probability_density(points, filename='density.png'):
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(points[:, 0], points[:, 1], gridsize=300, cmap='plasma', mincnt=1, linewidths=0)
    plt.title("Density Heatmap (Hexbin)")
    cb = plt.colorbar(hb)
    cb.set_label("Number of points")
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# --------------------------
# 4. Box-counting
# --------------------------

def box_counting_dimension(points, epsilons):
    N = []
    for eps in epsilons:
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
        bins_x = int(np.ceil((max_x - min_x) / eps))
        bins_y = int(np.ceil((max_y - min_y) / eps))
        H, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=(bins_x, bins_y))
        N.append(np.sum(H > 0))

    log_eps = np.log(1 / np.array(epsilons))
    log_N = np.log(N)
    slope, _, r_value, _, _ = linregress(log_eps, log_N)

    return slope, log_eps, log_N

def plot_box_dimension(log_eps, log_N, dim, filename='dimension.png'):
    plt.figure(figsize=(6, 4))
    plt.plot(log_eps, log_N, 'o-', label=f"Estimated dim ≈ {dim:.3f}")
    plt.xlabel("log(1/ε)")
    plt.ylabel("log(N(ε))")
    plt.title("Box-Counting Dimension Estimation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# --------------------------
# 5. Save points
# --------------------------

def save_points(points, filename='points.csv'):
    df = pd.DataFrame(points, columns=["x", "y"])
    df.to_csv(filename, index=False)

# --------------------------
# 6. Run single experiment
# --------------------------

def run_experiment(config, output_base="rnifs_outputs"):
    name = config["name"]
    print(f"▶ Running: {name}")

    output_dir = os.path.join(output_base, name)
    os.makedirs(output_dir, exist_ok=True)

    points = simulate_rnifs(
        functions=config["functions"],
        probabilities=config["probabilities"],
        num_points=config.get("num_points", 100000),
        burn_in=config.get("burn_in", 1000),
        seed=config.get("seed")
    )

    save_points(points, os.path.join(output_dir, f"{name}_points.csv"))
    plot_fractal(points, filename=os.path.join(output_dir, "fractal.png"), title=name)
    plot_probability_density(points, filename=os.path.join(output_dir, "density.png"))

    epsilons = np.logspace(-2, -0.3, num=10)
    dim, log_eps, log_N = box_counting_dimension(points, epsilons)
    plot_box_dimension(log_eps, log_N, dim, filename=os.path.join(output_dir, "dimension.png"))

    print(f"✔ Dimension ≈ {dim:.4f} | Output saved to: {output_dir}\n")

# --------------------------
# 7. Experiment definitions and runner
# --------------------------

if __name__ == "__main__":
    experiments = [
        
        {
            "name": "branching_structure",
            "functions": [f2, f5, f8],
            "probabilities": [0.5, 0.3, 0.2],
            "num_points": 100000,
            "burn_in": 2000,
            "seed": 101
        },
       
        {
            "name": "spiral_rotation",
            "functions": [f3, f7, f11],
            "probabilities": [0.4, 0.3, 0.3],
            "num_points": 150000,
            "burn_in": 2500,
            "seed": 202
        },
       
        {
            "name": "concentric_energy",
            "functions": [f1, f10, f12],
            "probabilities": [0.3, 0.3, 0.4],
            "num_points": 120000,
            "burn_in": 1800,
            "seed": 303
        },
   
        {
            "name": "chaotic_explosion",
            "functions": [f4, f6, f9, f11],
            "probabilities": [0.25, 0.25, 0.25, 0.25],
            "num_points": 200000,
            "burn_in": 3000,
            "seed": 404
        },

       
        {
            "name": "webbed_structure",
            "functions": [f3, f5, f7, f8],
            "probabilities": [0.25, 0.25, 0.25, 0.25],
            "num_points": 160000,
            "burn_in": 2500,
            "seed": 606
        },
       
        {
            "name": "disruptive_mixture",
            "functions": [f6, f9, f10],
            "probabilities": [0.6, 0.2, 0.2],
            "num_points": 100000,
            "burn_in": 1000,
            "seed": 707
        },
 
        {
            "name": "high_freq_disturbance",
            "functions": [f11, f12],
            "probabilities": [0.5, 0.5],
            "num_points": 90000,
            "burn_in": 1500,
            "seed": 808
        },

        
        {
            "name": "ultra_res_analysis",
            "functions": [f4, f5, f8],
            "probabilities": [0.3, 0.4, 0.3],
            "num_points": 300000,
            "burn_in": 5000,
            "seed": 1001
        }
    ]

    for config in experiments:
        run_experiment(config)
