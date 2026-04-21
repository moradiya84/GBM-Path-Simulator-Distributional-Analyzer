# GBM Path Simulator & Distributional Analyzer

A C++17 project that simulates **correlated multi-asset Geometric Brownian Motion (GBM)** using Monte Carlo methods, implements **Euler–Maruyama** and **Milstein** numerical schemes, computes **empirical distribution statistics**, benchmarks **strong convergence rates**, and exports results to CSV for Python visualization.

---

## Mathematical Model

Each asset follows GBM:

$$dS_t^{(i)} = \mu_i \, S_t^{(i)} \, dt + \sigma_i \, S_t^{(i)} \, dW_t^{(i)}$$

With correlated Brownian motions: $\text{Corr}(dW^{(i)}, dW^{(j)}) = \rho_{ij}$

### Exact Solution (Reference)

$$S_T^{(i)} = S_0^{(i)} \cdot \exp\!\left((\mu_i - \tfrac{1}{2}\sigma_i^2)T + \sigma_i W_T^{(i)}\right)$$

### Numerical Schemes

**Euler–Maruyama:**

$$S_{n+1} = S_n + \mu S_n \Delta t + \sigma S_n \Delta W_n$$

**Milstein:**

$$S_{n+1} = S_n + \mu S_n \Delta t + \sigma S_n \Delta W_n + \tfrac{1}{2}\sigma^2 S_n \left((\Delta W_n)^2 - \Delta t\right)$$

### Correlated Brownian Motion

At each time step: generate $z \sim \mathcal{N}(0, I)$, compute Cholesky factor $L$ of correlation matrix, then $\Delta W = \sqrt{\Delta t} \cdot L z$.

---

## Project Structure

```
├── include/
│   ├── gbm.hpp              # GBM model, config, exact solution
│   ├── sde_integrator.hpp    # Euler & Milstein stepping + path simulation
│   ├── cholesky.hpp          # Cholesky decomposition interface
│   ├── statistics.hpp        # Empirical & theoretical moment computations
│   └── convergence.hpp       # Convergence benchmark runner
├── src/
│   ├── cholesky.cpp          # Cholesky decomposition implementation
│   ├── statistics.cpp        # Mean, variance, skewness, kurtosis
│   └── convergence_runner.cpp# Strong convergence over multiple dt values
├── tests/
│   ├── test_integrator.cpp   # Euler step, path evolution, Milstein accuracy
│   ├── test_cholesky.cpp     # L*Lᵀ reconstruction, empirical correlation
│   └── test_statistics.cpp   # Known inputs, empirical vs theoretical GBM
├── scripts/
│   ├── plot_paths.py         # Visualize sample GBM trajectories
│   ├── plot_convergence.py   # Log-log strong convergence plot
│   └── plot_stats.py         # Empirical vs theoretical moment charts
├── data/                     # CSV outputs and generated plots
├── main.cpp                  # Full simulation driver
└── CMakeLists.txt
```

## Build

```bash
mkdir -p build && cd build
cmake ..
make
```

## Run

```bash
# From the build directory
./gbm_simulator

# Or from the project root
./build/gbm_simulator
```

## Run Tests

```bash
cd build
ctest --output-on-failure
```

Or individually:
```bash
./build/test_integrator
./build/test_cholesky
./build/test_statistics
```

## Python Plots

Requires `pandas` and `matplotlib`. Using a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas matplotlib
python3 scripts/plot_paths.py
python3 scripts/plot_convergence.py
python3 scripts/plot_stats.py
```

## CSV Outputs

All outputs are written to `data/`:

| File | Columns |
|---|---|
| `convergence.csv` | `dt`, `scheme`, `mean_error`, `rmse`, `log_dt`, `log_error` |
| `stats.csv` | `asset_id`, `mean_empirical`, `mean_theoretical`, `variance_empirical`, `variance_theoretical`, `skewness_empirical`, `kurtosis_empirical` |
| `paths.csv` | `path_id`, `time`, `asset_id`, `value` |

## Configuration

Simulation parameters are defined in `main.cpp` via the `GBMConfig` struct:

| Parameter | Description |
|---|---|
| `num_assets` | Number of correlated assets (M) |
| `num_paths` | Number of Monte Carlo paths (N) |
| `num_steps` | Time discretization steps |
| `T` | Maturity time |
| `S0` | Initial prices per asset |
| `mu` | Drift per asset |
| `sigma` | Volatility per asset |
| `correlation_matrix` | M × M correlation matrix |
| `scheme` | `Scheme::Euler` or `Scheme::Milstein` |

## Theoretical Moments (GBM)

$$\mathbb{E}[S_T] = S_0 \, e^{\mu T}$$

$$\text{Var}(S_T) = S_0^2 \, e^{2\mu T}\left(e^{\sigma^2 T} - 1\right)$$

## Convergence

Strong convergence is measured as:

$$\text{error} = \frac{1}{N}\sum_{i=1}^{N} |S_T^{\text{numerical}} - S_T^{\text{exact}}|$$

Expected convergence orders:
- **Euler–Maruyama**: O(Δt^0.5)
- **Milstein**: O(Δt^1.0)

## Results and Visualizations

The `data/` directory contains sample PNG plots generated from running the simulation described in `main.cpp`.

### 1. Simulated GBM Paths

**Input Configuration:**
- **Assets:** 3 correlated assets
- **Paths:** 10,000 Monte Carlo paths
- **Time Steps:** 100 steps over $T=1.0$
- **Initial Prices ($S_0$):** $\{100.0, 50.0, 75.0\}$
- **Drift ($\mu$):** $\{0.05, 0.08, 0.03\}$
- **Volatility ($\sigma$):** $\{0.2, 0.3, 0.15\}$
- **Correlation Matrix:**
  $$ \begin{bmatrix} 1.0 & 0.5 & 0.2 \\ 0.5 & 1.0 & -0.3 \\ 0.2 & -0.3 & 1.0 \end{bmatrix} $$

![Sample GBM Paths](data/paths_plot.png)

### 2. Empirical Distribution Statistics

Comparing the empirical mean, variance, skewness, and kurtosis of the simulated terminal prices against the theoretical exact moments. Data output is saved to `data/stats.csv`.

![Terminal Distribution Statistics](data/stats_plot.png)

### 3. Strong Convergence

**Input Configuration:**
- **Assets:** 1 asset
- **Paths:** 10,000 reference paths for exact solution comparison
- **Time Steps ($N$):** 10, 20, 50, 100, 200, 500 over $T=1.0$
- **Scheme:** Euler–Maruyama

The plot demonstrates the expected strong convergence rate for the selected numerical scheme ($\mathcal{O}(\Delta t^{0.5})$ for Euler). Data output is saved to `data/convergence.csv`.

![Strong Convergence Error](data/convergence_plot.png)
