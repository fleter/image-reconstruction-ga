import time
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

# External libraries:
# OpenCV is used for high-performance rendering of geometric shapes
# NumPy is used for matrix operations and fitness calculations
# Matplotlib is used for generating plots
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Configuration of main params of GA algorithm
@dataclass
class Config:
    image_size: int = 512

    # The grid into which we will divide the image (to use multiprocessing, compute each tile on 1 core)
    grid_rows: int = 8
    grid_cols: int = 8

    # Number of triangles per tile (more -> more detailed image, but slower computations)
    n_shapes: int = 80
    # Number of generations per tile (more -> a more similar result)
    generations: int = 300

    # Population size - number of candidates per generation
    population_size: int = 60
    # Probability of a gene mutation occurring
    mutation_rate: float = 0.05
    # Magnitude of mutation
    mutation_scale: float = 0.15
    # Elitism - number of the best individuals to carry over unchanged
    elite_size: int = 6
    # Number of individuals competing in each tournament selection round
    tournament_size: int = 3

    # Number of runs per image
    n_runs_global: int = 3

    # Number of k best images to save per one run
    k_top: int = 5

    # Max total runtime for the whole session in seconds (0 = no limit)
    max_runtime_sec: float = 300


# Draw the genome
def draw_individual(genome, height, width):
    # Initialize a black canvas (RGB channel, values from 0 to 255 (np.unit8))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Each gene is a triangle described by 10 parameters (R, G, B, Alpha, x1, y1, x2, y2, x3, y3)
    # All values are from 0.0 to 1.0 to make convenient to mutate and crossover,
    # and to not depend on from original size of the image
    for gene in genome:
        # Decode color
        r = int(gene[0] * 255)
        g = int(gene[1] * 255)
        b = int(gene[2] * 255)

        # Decode Alpha. Mapped to range 0.1 - 0.6, so triangle are never completely invisible or completely opaque
        alpha = gene[3] * 0.5 + 0.1

        # Decode coordinates (we get an array of three triangle vertices in pixels)
        pts = np.array([
            [int(gene[4] * width), int(gene[5] * height)],
            [int(gene[6] * width), int(gene[7] * height)],
            [int(gene[8] * width), int(gene[9] * height)]
        ], np.int32).reshape((-1, 1, 2))  # Made for proper work of OpenCVz

        # Draw the triangle using OpenCV
        overlay = canvas.copy()
        color = (b, g, r)  # OpenCV work with BGR format
        cv2.fillPoly(overlay, [pts], color)

        # Alpha blending to create transparency effects
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    # Return all drawn triangles from genome
    return canvas


# Calculate Root Mean Squared Error between genome and original image
# We use RMSE because this metric shows the error in the same units as the pixel colors (scale 0–255),
# making the result intuitive.
def calculate_fitness(genome, target_img):
    h, w, _ = target_img.shape
    generated = draw_individual(genome, h, w)

    mse = np.mean((generated.astype(np.float32) - target_img.astype(np.float32)) ** 2)
    rmse = np.sqrt(mse)

    # Return a negative value because the GA maximizes fitness, but we want to minimize error.
    return -rmse


# Initializes a random genome - matrix with n triangles with 10 params (RGB, Alpha, coords of vertices)
def create_genome(n_shapes):
    return np.random.rand(n_shapes, 10).astype(np.float32)


# Performs Uniform Crossover between two parents
# For each gene, it is randomly chosen whether to take it from the first parent or from the second
def crossover(p1, p2):
    mask = np.random.rand(*p1.shape) < 0.5
    return np.where(mask, p1, p2)


# Applies Gaussian Mutation to the genome
def mutation(genome, rate, scale):
    # With probability rate, random noise from a normal distribution is added to each gene
    mask = np.random.rand(*genome.shape) < rate
    noise = np.random.normal(0, scale, size=genome.shape)
    genome[mask] += noise[mask]
    # And values are truncated to the range [0.0, 1.0]
    return np.clip(genome, 0.0, 1.0)


# Worker function for a single image title
# Runs in parallel processes
def worker_title(tile_data):
    target_img, cfg_dict, seed, tile_idx = tile_data

    # Unpack configuration
    pop_size = cfg_dict['population_size']
    generations = cfg_dict['generations']
    n_shapes = cfg_dict['n_shapes']
    mut_rate = cfg_dict['mutation_rate']
    mut_scale = cfg_dict['mutation_scale']
    elite_size = cfg_dict['elite_size']
    tourn_k = cfg_dict['tournament_size']
    # How many best genomes per tile we want to export (for global top-k reconstructions)
    k_top = cfg_dict.get('k_top', 5)

    np.random.seed(seed)
    h, w, _ = target_img.shape

    # Initialization of population and calculate fitness for each genome
    population = [create_genome(n_shapes) for _ in range(pop_size)]
    fitnesses = [calculate_fitness(g, target_img) for g in population]

    # Sort population by fitness
    pop_fit = list(zip(population, fitnesses))
    pop_fit.sort(key=lambda x: x[1], reverse=True)
    population = [x[0] for x in pop_fit]
    fitnesses = [x[1] for x in pop_fit]

    # History for plotting
    history_best = []
    history_avg = []

    # Evolution loop
    for gen in range(generations):
        # Write statistic of this generation
        current_best = fitnesses[0]
        current_avg = sum(fitnesses) / len(fitnesses)
        # And save to history
        history_best.append(current_best)
        history_avg.append(current_avg)

        # Elitism (keep only the best)
        new_pop = []
        new_pop.extend([p.copy() for p in population[:elite_size]])

        # Generation of offspring until the population is full
        while len(new_pop) < pop_size:
            # Tournament selection
            # For each parent choose tourn_k random indexes, check fitness, choose the best and add genome to parents
            cands_idx = np.random.randint(0, pop_size, size=(2, tourn_k))
            parents = []
            for k in range(2):
                cands = cands_idx[k]
                best_idx = cands[0]
                best_val = fitnesses[best_idx]
                for idx in cands[1:]:
                    if fitnesses[idx] > best_val:
                        best_val = fitnesses[idx]
                        best_idx = idx
                parents.append(population[best_idx])

            # Crossover and mutation
            child = crossover(parents[0], parents[1])
            child = mutation(child, mut_rate, mut_scale)
            new_pop.append(child)

        # Population update and recalculation of fitnesses
        population = new_pop
        fitnesses = [calculate_fitness(g, target_img) for g in population]

        # Sort again
        pop_fit = list(zip(population, fitnesses))
        pop_fit.sort(key=lambda x: x[1], reverse=True)
        population = [x[0] for x in pop_fit]
        fitnesses = [x[1] for x in pop_fit]

    # Final Result:
    # Take top-k genomes for this tile
    top_genomes = population[:k_top]
    top_fits = fitnesses[:k_top]

    # Draw k_top versions of this tile (for each of the top-k genomes)
    top_tiles = [draw_individual(g, h, w) for g in top_genomes]

    # Return index of tile, list of k_top tiles, list of k_top fitnesses and statistics history
    return (tile_idx, top_tiles, top_fits, history_best, history_avg)


# Main GA function
# Manages the tiling, multiprocessing, and construction of the final image
def ga_launch(img_path, cfg, run_id, deadline=None):
    # Load and Resize Image
    original = cv2.imread(str(img_path))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (cfg.image_size, cfg.image_size))

    h, w, _ = original.shape
    tile_h = h // cfg.grid_rows
    tile_w = w // cfg.grid_cols

    # Prepare tasks for parallel workers
    tasks = []
    cfg_dict = vars(cfg)  # Convert dataclass to dict for child processes

    # Loop on grid
    counter = 0  # Index of tile
    for r in range(cfg.grid_rows):
        for c in range(cfg.grid_cols):
            y1, y2 = r * tile_h, (r + 1) * tile_h
            x1, x2 = c * tile_w, (c + 1) * tile_w
            tile_img = original[y1:y2, x1:x2]
            # Unique seed for each tile
            seed = run_id * 100000 + counter
            tasks.append((tile_img, cfg_dict, seed, counter))
            counter += 1

    cpu_count = mp.cpu_count()
    k_top = cfg.k_top  # How many global best reconstructions we will build

    # For each tile we keep k_top variants
    results_storage = [[None] * k_top for _ in range(counter)]

    print(f"[Run {run_id}] Launching {counter} tiles with {cpu_count} CPUs")
    start_t = time.time()

    # Arrays for global statistics
    global_history_best = np.zeros(cfg.generations)
    global_history_avg = np.zeros(cfg.generations)

    finished_count = 0
    total_rmse = 0  # Here we accumulate the best fitness values (negative RMSE)

    # Execute parallel processing
    time_exceeded = False
    pool = mp.Pool(processes=cpu_count)
    try:
        # imap_unordered allows us to see results as they finish
        for tile_idx, tile_imgs, tile_fits, hist_best, hist_avg in pool.imap_unordered(worker_title, tasks):
            # Save k_top tiles for this tile index
            results_storage[tile_idx] = tile_imgs
            finished_count += 1

            # Accumulate stats from this tile
            global_history_best += np.array(hist_best)
            global_history_avg += np.array(hist_avg)

            # We track the best fitness per tile for logging (0 - best, fitness = -RMSE)
            best_fit = tile_fits[0]
            total_rmse += best_fit

            # Calculate grid coordinates
            row = tile_idx // cfg.grid_cols
            col = tile_idx % cfg.grid_cols

            # Print progress
            print(f"[Tile {row},{col}] Finished. Fitness (best): {best_fit:6.2f} | Progress: {finished_count}/{counter}")

            if deadline is not None and time.time() > deadline:
                time_exceeded = True
                print("  [Time limit] Reached. Stopping early and returning partial result.")
                pool.terminate()
                break
    finally:
        if not time_exceeded:
            pool.close()
        pool.join()

    elapsed = time.time() - start_t
    print(f"  [Run {run_id}] All tiles finished in {elapsed:.2f}s")

    # Average the statistics across finished tiles
    if finished_count > 0:
        global_history_best /= finished_count
        global_history_avg /= finished_count
        avg_final_rmse = total_rmse / finished_count  # Average best fitness per tile (still negative RMSE)
        print(f"  [Run {run_id}] Average best fitness per tile: {avg_final_rmse:.2f}")
    else:
        print(f"  [Run {run_id}] No tiles finished before time limit.")

    # Reconstruct k_top final images
    final_canvases = []
    fallback_tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    for rank in range(k_top):
        final_canvas = np.zeros_like(original)
        idx = 0
        for r in range(cfg.grid_rows):
            for c in range(cfg.grid_cols):
                # For each tile take tile image with given rank (0 - best, 1 - second best, ...)
                tile_entry = results_storage[idx]
                tile_res = tile_entry[rank] if tile_entry[rank] is not None else fallback_tile
                idx += 1
                y1, y2 = r * tile_h, (r + 1) * tile_h
                x1, x2 = c * tile_w, (c + 1) * tile_w
                final_canvas[y1:y2, x1:x2] = tile_res
        final_canvases.append(final_canvas)

    # Plot generation
    plt.figure(figsize=(10, 6))
    x_axis = range(cfg.generations)
    plt.plot(x_axis, global_history_best, label='Best Fitness', color='blue')
    plt.plot(x_axis, global_history_avg, label='Average Fitness', color='orange', linestyle='--')
    plt.xlabel('Generations')
    plt.ylabel('Fitness (-RMSE)')
    plt.title(f'Plot - Run {run_id} - {img_path.name}')
    plt.legend()
    plt.grid(True)

    plot_filename = f"{img_path.stem}_plot_run{run_id}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"  [Stats] Saved plot to {plot_filename}")

    # Return list of k_top final reconstructed images for this run
    return final_canvases


def main():
    cfg = Config()
    input_dir = Path(".")
    inputs = list(input_dir.glob("input*.jpg"))

    total_start = time.time()
    deadline = total_start + cfg.max_runtime_sec if cfg.max_runtime_sec > 0 else None

    for img_path in inputs:
        print(f"\n Processing {img_path.name}")
        for i in range(1, cfg.n_runs_global + 1):
            if deadline is not None and time.time() > deadline:
                print("Time limit reached. Stopping before starting a new run.")
                break
            # For each run we now get k_top rebuild images
            result_images = ga_launch(img_path, cfg, i, deadline=deadline)

            # Save each of k_top best reconstructions separately
            for rank, result_image in enumerate(result_images, start=1):
                out_name = input_dir / f"{img_path.stem}_result_run{i}_top{rank}.png"
                cv2.imwrite(str(out_name), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                print(f"  [Output] Saved image to {out_name}")

            if deadline is not None and time.time() > deadline:
                print("Time limit reached. Stopping after current partial results.")
                break

    print(f"\nTotal Session Time: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()

