import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Dict
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BenchmarkFunctions:
    """benchmark optimization functions"""

    @staticmethod
    def sphere(x):
        return np.sum(x ** 2)

    @staticmethod
    def rastrigin(x):
        A = 10
        return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

    @staticmethod
    def ackley(x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        n = len(x)
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(c * x))
        term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)
        return term1 + term2 + a + np.exp(1)

    @staticmethod
    def get_function(name: str) -> Callable:
        functions = {
            'sphere': BenchmarkFunctions.sphere,
            'rastrigin': BenchmarkFunctions.rastrigin,
            'ackley': BenchmarkFunctions.ackley
        }
        return functions.get(name, BenchmarkFunctions.sphere)


class GeneticAlgorithm:
    """GA"""

    def __init__(self, objective_func, dim=2, bounds=(-5.12, 5.12)):
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = bounds

    def run(self, max_evaluations=2000, pop_size=50, crossover_rate=0.8,
            mutation_rate=0.1, tournament_size=3):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (pop_size, self.dim))
        fitness = np.array([self.objective_func(ind) for ind in population])
        evaluations = pop_size
        best_fitness = np.min(fitness)

        while evaluations < max_evaluations:
            # Tournament selection
            parents = []
            for _ in range(pop_size):
                contestants = np.random.choice(pop_size, tournament_size, replace=False)
                winner = contestants[np.argmin(fitness[contestants])]
                parents.append(population[winner])
            parents = np.array(parents)

            # Crossover (uniform)
            offspring = parents.copy()
            for i in range(0, pop_size, 2):
                if i + 1 < pop_size and np.random.random() < crossover_rate:
                    mask = np.random.random(self.dim) < 0.5
                    offspring[i][mask] = parents[i + 1][mask]
                    offspring[i + 1][mask] = parents[i][mask]

            # Mutation (Gaussian)
            for i in range(pop_size):
                if np.random.random() < mutation_rate:
                    mutation_strength = (self.bounds[1] - self.bounds[0]) * 0.1
                    offspring[i] += np.random.normal(0, mutation_strength, self.dim)
                    offspring[i] = np.clip(offspring[i], self.bounds[0], self.bounds[1])

            # Evaluate offspring
            new_fitness = np.array([self.objective_func(ind) for ind in offspring])
            evaluations += pop_size

            # Elitism: keep best from previous generation
            best_idx = np.argmin(fitness)
            worst_idx = np.argmax(new_fitness)
            offspring[worst_idx] = population[best_idx]
            new_fitness[worst_idx] = fitness[best_idx]

            population, fitness = offspring, new_fitness
            current_best = np.min(fitness)
            best_fitness = min(best_fitness, current_best)

        return best_fitness


class DifferentialEvolution:
    """DE"""

    def __init__(self, objective_func, dim=2, bounds=(-5.12, 5.12)):
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = bounds

    def run(self, max_evaluations=2000, pop_size=50, F=0.5, CR=0.7):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (pop_size, self.dim))
        fitness = np.array([self.objective_func(ind) for ind in population])
        evaluations = pop_size
        best_fitness = np.min(fitness)

        while evaluations < max_evaluations:
            for i in range(pop_size):
                # Select three distinct random individuals
                candidates = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(candidates, 3, replace=False)]

                # Mutation and crossover
                mutant = a + F * (b - c)
                trial = population[i].copy()

                # Binomial crossover
                cross_points = np.random.random(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial[cross_points] = mutant[cross_points]
                trial = np.clip(trial, self.bounds[0], self.bounds[1])

                # Selection
                trial_fitness = self.objective_func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    best_fitness = min(best_fitness, trial_fitness)

                if evaluations >= max_evaluations:
                    break

        return best_fitness


class InnerPSO:
    """Inner PSO to be fine tuned"""

    def __init__(self, objective_func, dim=2, bounds=(-5.12, 5.12)):
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = bounds

    def run(self, max_evaluations=2000, pop_size=30, inertia=0.7,
            cognitive_weight=1.5, social_weight=1.5):
        # Initialize particles and velocities
        particles = np.random.uniform(self.bounds[0], self.bounds[1], (pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (pop_size, self.dim))

        personal_best_positions = particles.copy()
        personal_best_scores = np.array([self.objective_func(p) for p in particles])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = pop_size

        while evaluations < max_evaluations:
            for i in range(pop_size):
                # Update velocity
                r1, r2 = np.random.random(2)
                cognitive_component = cognitive_weight * r1 * (personal_best_positions[i] - particles[i])
                social_component = social_weight * r2 * (global_best_position - particles[i])
                velocities[i] = inertia * velocities[i] + cognitive_component + social_component

                # Update position
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.bounds[0], self.bounds[1])

                # Evaluate
                current_fitness = self.objective_func(particles[i])
                evaluations += 1

                # Update personal best
                if current_fitness < personal_best_scores[i]:
                    personal_best_positions[i] = particles[i].copy()
                    personal_best_scores[i] = current_fitness

                    # Update global best
                    if current_fitness < global_best_score:
                        global_best_position = particles[i].copy()
                        global_best_score = current_fitness

                if evaluations >= max_evaluations:
                    break

        return global_best_score


class MetaPSO:
    """Meta PSO that tunes the parameters of other algorithms"""

    def __init__(self, test_function_name='rastrigin', dim=2):
        self.test_function = BenchmarkFunctions.get_function(test_function_name)
        self.dim = dim
        self.bounds = (-5.12, 5.12)  # Standard bounds for benchmark functions

        # Algorithm instances
        self.ga = GeneticAlgorithm(self.test_function, dim, self.bounds)
        self.de = DifferentialEvolution(self.test_function, dim, self.bounds)
        self.inner_pso = InnerPSO(self.test_function, dim, self.bounds)

        # Define parameter bounds for each algorithm type
        self.algorithm_bounds = {
            'GA': [
                (20, 100),  # pop_size
                (0.5, 0.95),  # crossover_rate
                (0.01, 0.2)  # mutation_rate
            ],
            'DE': [
                (20, 100),  # pop_size
                (0.2, 0.9),  # F
                (0.5, 0.99)  # CR
            ],
            'PSO': [
                (20, 100),  # pop_size
                (0.4, 0.9),  # inertia
                (1.0, 2.5),  # cognitive_weight
                (1.0, 2.5)  # social_weight
            ]
        }

    def evaluate_parameters(self, algorithm_type: str, parameters: List[float],
                            max_evaluations: int = 1000, num_runs: int = 3) -> float:
        """Evaluate a parameter set by running the algorithm multiple times"""
        total_performance = 0

        for run in range(num_runs):
            if algorithm_type == 'GA':
                pop_size = int(parameters[0])
                crossover_rate = parameters[1]
                mutation_rate = parameters[2]
                performance = self.ga.run(
                    max_evaluations=max_evaluations,
                    pop_size=pop_size,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate
                )

            elif algorithm_type == 'DE':
                pop_size = int(parameters[0])
                F = parameters[1]
                CR = parameters[2]
                performance = self.de.run(
                    max_evaluations=max_evaluations,
                    pop_size=pop_size,
                    F=F,
                    CR=CR
                )

            elif algorithm_type == 'PSO':
                pop_size = int(parameters[0])
                inertia = parameters[1]
                cognitive_weight = parameters[2]
                social_weight = parameters[3]
                performance = self.inner_pso.run(
                    max_evaluations=max_evaluations,
                    pop_size=pop_size,
                    inertia=inertia,
                    cognitive_weight=cognitive_weight,
                    social_weight=social_weight
                )

            total_performance += performance

        return total_performance / num_runs  # Average performance

    def run_meta_optimization(self, algorithm_type: str,
                              meta_pop_size: int = 10,
                              meta_generations: int = 20,
                              max_evaluations: int = 1000) -> Dict:
        """Run the meta-PSO to optimize parameters for a specific algorithm"""

        bounds = self.algorithm_bounds[algorithm_type]
        dim = len(bounds)

        # Initialize meta-PSO particles
        particles = np.random.uniform(0, 1, (meta_pop_size, dim))
        velocities = np.random.uniform(-0.1, 0.1, (meta_pop_size, dim))

        # Scale particles to parameter bounds
        def scale_particle(particle):
            scaled = []
            for i, (low, high) in enumerate(bounds):
                scaled.append(low + particle[i] * (high - low))
            return scaled

        # Initialize personal and global bests
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(meta_pop_size, np.inf)

        for i in range(meta_pop_size):
            scaled_params = scale_particle(particles[i])
            fitness = self.evaluate_parameters(algorithm_type, scaled_params, max_evaluations, num_runs=2)
            personal_best_scores[i] = fitness

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = particles[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

        # PSO parameters for meta-optimization
        inertia = 0.7
        cognitive_weight = 1.5
        social_weight = 1.5

        convergence_history = [global_best_score]

        logging.info(f"Starting meta-optimization for {algorithm_type}")
        logging.info(f"Initial best fitness: {global_best_score:.6f}")

        for generation in range(meta_generations):
            for i in range(meta_pop_size):
                # Update velocity
                r1, r2 = np.random.random(2)
                cognitive = cognitive_weight * r1 * (personal_best_positions[i] - particles[i])
                social = social_weight * r2 * (global_best_position - particles[i])
                velocities[i] = inertia * velocities[i] + cognitive + social

                # Update position
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], 0, 1)

                # Evaluate
                scaled_params = scale_particle(particles[i])
                fitness = self.evaluate_parameters(algorithm_type, scaled_params, max_evaluations, num_runs=2)

                # Update personal best
                if fitness < personal_best_scores[i]:
                    personal_best_positions[i] = particles[i].copy()
                    personal_best_scores[i] = fitness

                    # Update global best
                    if fitness < global_best_score:
                        global_best_position = particles[i].copy()
                        global_best_score = fitness
                        logging.info(f"Generation {generation}: New best fitness: {global_best_score:.6f}")

            convergence_history.append(global_best_score)

        # Get the best parameters
        best_scaled_params = scale_particle(global_best_position)

        # Final evaluation with more runs for accuracy
        final_performance = self.evaluate_parameters(
            algorithm_type, best_scaled_params, max_evaluations, num_runs=5
        )

        result = {
            'best_parameters': best_scaled_params,
            'best_fitness': final_performance,
            'convergence_history': convergence_history,
            'algorithm_type': algorithm_type
        }

        return result


def main():
    """Main demonstration function"""
    # Create meta-optimizer
    meta_optimizer = MetaPSO(test_function_name='rastrigin', dim=5)

    # Test algorithms to optimize
    algorithms = ['GA', 'DE', 'PSO']

    results = {}

    # Run meta-optimization for each algorithm
    for algo in algorithms:
        print(f"\n{'=' * 50}")
        print(f"Optimizing parameters for {algo}")
        print(f"{'=' * 50}")

        result = meta_optimizer.run_meta_optimization(
            algorithm_type=algo,
            meta_pop_size=8,  # Small for demonstration
            meta_generations=10,
            max_evaluations=800
        )

        results[algo] = result

        print(f"Best parameters for {algo}:")
        param_names = {
            'GA': ['pop_size', 'crossover_rate', 'mutation_rate'],
            'DE': ['pop_size', 'F', 'CR'],
            'PSO': ['pop_size', 'inertia', 'cognitive_weight', 'social_weight']
        }

        for name, value in zip(param_names[algo], result['best_parameters']):
            if name == 'pop_size':
                value = int(value)
            print(f"  {name}: {value:.4f}")
        print(f"Best fitness: {result['best_fitness']:.6f}")

    # Plot convergence histories
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        history = results[algo]['convergence_history']
        plt.plot(history, label=f'{algo}', linewidth=2)

    plt.xlabel('Meta-Generation')
    plt.ylabel('Best Fitness (Lower is Better)')
    plt.title('Meta-PSO Convergence History\n(Tuning Algorithm Parameters)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    # Compare with default parameters
    print(f"\n{'=' * 50}")
    print("COMPARISON WITH DEFAULT PARAMETERS")
    print(f"{'=' * 50}")

    default_params = {
        'GA': [50, 0.8, 0.1],
        'DE': [50, 0.5, 0.7],
        'PSO': [30, 0.7, 1.5, 1.5]
    }

    for algo in algorithms:
        default_perf = meta_optimizer.evaluate_parameters(
            algo, default_params[algo], max_evaluations=800, num_runs=5
        )
        tuned_perf = results[algo]['best_fitness']

        improvement = ((default_perf - tuned_perf) / default_perf) * 100

        print(f"\n{algo}:")
        print(f"  Default: {default_perf:.6f}")
        print(f"  Tuned:   {tuned_perf:.6f}")
        print(f"  Improvement: {improvement:+.2f}%")


if __name__ == "__main__":
    main()