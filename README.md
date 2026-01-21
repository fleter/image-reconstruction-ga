# Image Reconstruction using Genetic Algorithms

This project implements a **distributed genetic algorithm** for image reconstruction using **semi-transparent triangles**.  
The goal is to approximate a target image under strict time constraints while producing a stylized *low-poly* visual result.

The algorithm uses a **tile-based divide-and-conquer approach** to significantly reduce computational complexity and fully utilize multi-core CPUs via parallel processing.

---

## Key Features

- Tile-based **distributed genetic algorithm**
- Parallel processing using Python `multiprocessing`
- Image approximation with **semi-transparent triangles**
- Fitness based on **negative RMSE**
- OpenCV-based high-performance rendering
- Stable convergence with elitism and tournament selection
- Statistical analysis and convergence plots

---

## Methodology

### 1. Image Tiling

- Input image is resized to **512 × 512**
- Divided into an **8 × 8 grid** (64 tiles of 64 × 64 pixels)
- Each tile is processed independently on a separate CPU core

This reduces the search space and allows near-100% CPU utilization.

---

### 2. Genome Representation

Each individual represents a single tile and consists of a fixed number of triangles.

- **80 triangles per tile**
- Each triangle is encoded by **10 genes**: [R, G, B, α, x1, y1, x2, y2, x3, y3]

Where:
- `R, G, B` — color channels  
- `α` — transparency (mapped to range 0.1–0.6)  
- `(x, y)` — triangle vertex coordinates (normalized)

---

### 3. Fitness Function

Fitness is computed as **negative Root Mean Squared Error (RMSE)** between the generated tile and the target tile:

- RMSE provides interpretable pixel-scale error
- Fitness is maximized (lower error → higher fitness)

A perfect reconstruction corresponds to a fitness of `0` (but this happens very rarely tbh)

---

### 4. Genetic Operators

- **Selection:** Tournament selection (k = 3)
- **Crossover:** Uniform crossover (50% probability per gene)
- **Mutation:** Gaussian mutation  
  - rate = 0.05  
  - scale (σ) = 0.15
- **Elitism:** Top 6 individuals preserved each generation

---

## Configuration Parameters
All parameters can be easily changed via Config secton in the code,
but there are no guarantees that the code will work stably if you changed something

| Parameter | Value |
|--------|-------|
| Image size | 512 × 512 |
| Grid size | 8 × 8 (64 tiles) |
| Population size | 60 |
| Generations | 300 |
| Shapes per tile | 80 |
| Mutation rate | 0.05 |
| Mutation scale | 0.15 |
| Elite size | 6 |
| Tournament size | 3 |
| Runs per image | 3 |

---

## Results

- Stable convergence across multiple independent runs
- Consistent final RMSE values
- Produces visually pleasing *low-poly* reconstructions
- Maintains genetic diversity throughout evolution

The tiling structure remains visible in the final image, contributing to the artistic mosaic effect.

---

## How to Run

### Requirements

- Python 3.9+
- OpenCV
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

## Execution

Place input images in the project root with names:
    ```
    input1.jpg
    input2.jpg
    ```

Run:

    python GA.py

The script will:
- process each image
- run multiple independent experiments
- save reconstructed images and convergence plots

## Background

This project was developed as a university assignment focused on evolutionary algorithms.

The implementation was developed locally and later published on GitHub in its finalized form.


