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


class HybridAlgorithm:
    """Hybrid algorithm that combines DE, PSO, and GA"""

    def __init__(self, objective_func, dim=2, bounds=(-5.12, 5.12)):
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = bounds

    def run_hybrid(self, max_evaluations=2000,
                   # DE parameters
                   de_pop_size=30, de_F=0.5, de_CR=0.7,
                   # PSO parameters
                   pso_pop_size=20, pso_inertia=0.7, pso_cognitive=1.5, pso_social=1.5,
                   # GA parameters
                   ga_pop_size=25, ga_crossover_rate=0.8, ga_mutation_rate=0.1,
                   # Hybrid weights (how much each algorithm contributes)
                   de_weight=0.33, pso_weight=0.33, ga_weight=0.34):
        """
        Run hybrid algorithm where all three algorithms work together
        and their solutions are combined based on weights
        """
        total_pop_size = de_pop_size + pso_pop_size + ga_pop_size
        evaluations = 0

        # Initialize populations for each algorithm
        de_population = np.random.uniform(self.bounds[0], self.bounds[1], (de_pop_size, self.dim))
        pso_population = np.random.uniform(self.bounds[0], self.bounds[1], (pso_pop_size, self.dim))
        ga_population = np.random.uniform(self.bounds[0], self.bounds[1], (ga_pop_size, self.dim))

        # Initialize velocities for PSO
        pso_velocities = np.random.uniform(-1, 1, (pso_pop_size, self.dim))
        pso_personal_best_positions = pso_population.copy()
        pso_personal_best_scores = np.array([self.objective_func(p) for p in pso_population])
        pso_global_best_idx = np.argmin(pso_personal_best_scores)
        pso_global_best_position = pso_personal_best_positions[pso_global_best_idx].copy()

        # Evaluate initial populations
        de_fitness = np.array([self.objective_func(ind) for ind in de_population])
        ga_fitness = np.array([self.objective_func(ind) for ind in ga_population])

        evaluations += total_pop_size

        best_fitness = min(np.min(de_fitness), np.min(ga_fitness), np.min(pso_personal_best_scores))
        best_solution = None

        # Track best solutions from each algorithm
        de_best_idx = np.argmin(de_fitness)
        ga_best_idx = np.argmin(ga_fitness)

        iteration = 0

        while evaluations < max_evaluations:
            iteration += 1

            # 1. Run DE iteration
            for i in range(de_pop_size):
                if evaluations >= max_evaluations:
                    break

                # Select three distinct random individuals
                candidates = [idx for idx in range(de_pop_size) if idx != i]
                a, b, c = de_population[np.random.choice(candidates, 3, replace=False)]

                # Mutation and crossover
                mutant = a + de_F * (b - c)
                trial = de_population[i].copy()

                # Binomial crossover
                cross_points = np.random.random(self.dim) < de_CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial[cross_points] = mutant[cross_points]
                trial = np.clip(trial, self.bounds[0], self.bounds[1])

                # Selection
                trial_fitness = self.objective_func(trial)
                evaluations += 1

                if trial_fitness < de_fitness[i]:
                    de_population[i] = trial
                    de_fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial.copy()

            # 2. Run PSO iteration
            for i in range(pso_pop_size):
                if evaluations >= max_evaluations:
                    break

                # Update velocity
                r1, r2 = np.random.random(2)
                cognitive_component = pso_cognitive * r1 * (pso_personal_best_positions[i] - pso_population[i])
                social_component = pso_social * r2 * (pso_global_best_position - pso_population[i])
                pso_velocities[i] = pso_inertia * pso_velocities[i] + cognitive_component + social_component

                # Update position
                pso_population[i] += pso_velocities[i]
                pso_population[i] = np.clip(pso_population[i], self.bounds[0], self.bounds[1])

                # Evaluate
                current_fitness = self.objective_func(pso_population[i])
                evaluations += 1

                # Update personal best
                if current_fitness < pso_personal_best_scores[i]:
                    pso_personal_best_positions[i] = pso_population[i].copy()
                    pso_personal_best_scores[i] = current_fitness

                    # Update global best
                    if current_fitness < self.objective_func(pso_global_best_position):
                        pso_global_best_position = pso_population[i].copy()
                        pso_global_best_idx = i

                    if current_fitness < best_fitness:
                        best_fitness = current_fitness
                        best_solution = pso_population[i].copy()

            # 3. Run GA iteration
            if evaluations < max_evaluations:
                # Tournament selection
                parents = []
                for _ in range(ga_pop_size):
                    contestants = np.random.choice(ga_pop_size, 3, replace=False)
                    winner = contestants[np.argmin(ga_fitness[contestants])]
                    parents.append(ga_population[winner])
                parents = np.array(parents)

                # Crossover (uniform)
                offspring = parents.copy()
                for i in range(0, ga_pop_size, 2):
                    if i + 1 < ga_pop_size and np.random.random() < ga_crossover_rate:
                        mask = np.random.random(self.dim) < 0.5
                        offspring[i][mask] = parents[i + 1][mask]
                        offspring[i + 1][mask] = parents[i][mask]

                # Mutation (Gaussian)
                for i in range(ga_pop_size):
                    if np.random.random() < ga_mutation_rate:
                        mutation_strength = (self.bounds[1] - self.bounds[0]) * 0.1
                        offspring[i] += np.random.normal(0, mutation_strength, self.dim)
                        offspring[i] = np.clip(offspring[i], self.bounds[0], self.bounds[1])

                # Evaluate offspring
                new_fitness = np.array([self.objective_func(ind) for ind in offspring])
                evaluations += ga_pop_size

                # Elitism: keep best from previous generation
                best_idx = np.argmin(ga_fitness)
                worst_idx = np.argmax(new_fitness)
                offspring[worst_idx] = ga_population[best_idx]
                new_fitness[worst_idx] = ga_fitness[best_idx]

                ga_population, ga_fitness = offspring, new_fitness

                current_best = np.min(ga_fitness)
                if current_best < best_fitness:
                    best_fitness = current_best
                    best_solution = ga_population[np.argmin(ga_fitness)].copy()

            # 4. Information exchange between algorithms (optional)
            # Every few iterations, share best solutions
            if iteration % 5 == 0 and evaluations < max_evaluations:
                self._exchange_information(de_population, de_fitness,
                                           pso_population, pso_personal_best_scores,
                                           ga_population, ga_fitness,
                                           de_weight, pso_weight, ga_weight)

        return best_fitness

    def _exchange_information(self, de_pop, de_fit, pso_pop, pso_fit, ga_pop, ga_fit,
                              de_weight, pso_weight, ga_weight):
        """Exchange best solutions between algorithms"""
        # Find best individuals from each algorithm
        de_best_idx = np.argmin(de_fit)
        pso_best_idx = np.argmin(pso_fit)
        ga_best_idx = np.argmin(ga_fit)

        de_best = de_pop[de_best_idx]
        pso_best = pso_pop[pso_best_idx]
        ga_best = ga_pop[ga_best_idx]

        # Replace worst individuals in each population with best from other algorithms
        # Based on the weights, more successful algorithms contribute more

        # Replace in DE population
        if pso_weight > de_weight:
            de_worst_idx = np.argmax(de_fit)
            de_pop[de_worst_idx] = pso_best.copy()
        if ga_weight > de_weight:
            de_worst_idx = np.argmax(de_fit)
            de_pop[de_worst_idx] = ga_best.copy()

        # Replace in PSO population
        if de_weight > pso_weight:
            pso_worst_idx = np.argmax(pso_fit)
            pso_pop[pso_worst_idx] = de_best.copy()
        if ga_weight > pso_weight:
            pso_worst_idx = np.argmax(pso_fit)
            pso_pop[pso_worst_idx] = ga_best.copy()

        # Replace in GA population
        if de_weight > ga_weight:
            ga_worst_idx = np.argmax(ga_fit)
            ga_pop[ga_worst_idx] = de_best.copy()
        if pso_weight > ga_weight:
            ga_worst_idx = np.argmax(ga_fit)
            ga_pop[ga_worst_idx] = pso_best.copy()


class HybridMetaPSO:
    """Meta PSO that tunes ALL parameters of the hybrid algorithm"""

    def __init__(self, test_function_name='rastrigin', dim=2):
        self.test_function = BenchmarkFunctions.get_function(test_function_name)
        self.dim = dim

        # Hybrid algorithm instance
        self.hybrid_algo = HybridAlgorithm(self.test_function, dim)

        # Define ALL parameter bounds for the hybrid algorithm
        self.hybrid_bounds = [
            # DE parameters
            (10, 50),  # de_pop_size
            (0.1, 0.9),  # de_F
            (0.3, 0.99),  # de_CR

            # PSO parameters
            (10, 40),  # pso_pop_size
            (0.3, 0.9),  # pso_inertia
            (0.5, 2.5),  # pso_cognitive
            (0.5, 2.5),  # pso_social

            # GA parameters
            (10, 40),  # ga_pop_size
            (0.5, 0.95),  # ga_crossover_rate
            (0.01, 0.3),  # ga_mutation_rate

            # Hybrid weights (must sum to ~1.0)
            (0.1, 0.6),  # de_weight
            (0.1, 0.6),  # pso_weight
            # ga_weight is computed as 1.0 - de_weight - pso_weight
        ]

    def evaluate_hybrid_parameters(self, parameters: List[float],
                                   max_evaluations: int = 1500,
                                   num_runs: int = 3) -> float:
        """Evaluate hybrid algorithm with given parameters"""
        total_performance = 0

        for run in range(num_runs):
            # Extract and validate parameters
            de_pop_size = int(parameters[0])
            de_F = parameters[1]
            de_CR = parameters[2]

            pso_pop_size = int(parameters[3])
            pso_inertia = parameters[4]
            pso_cognitive = parameters[5]
            pso_social = parameters[6]

            ga_pop_size = int(parameters[7])
            ga_crossover_rate = parameters[8]
            ga_mutation_rate = parameters[9]

            de_weight = parameters[10]
            pso_weight = parameters[11]
            ga_weight = 1.0 - de_weight - pso_weight

            # Ensure weights are valid
            if ga_weight < 0.1:  # Minimum weight for any algorithm
                ga_weight = 0.1
                # Re-normalize
                total = de_weight + pso_weight + ga_weight
                de_weight /= total
                pso_weight /= total
                ga_weight /= total

            performance = self.hybrid_algo.run_hybrid(
                max_evaluations=max_evaluations,
                # DE parameters
                de_pop_size=de_pop_size,
                de_F=de_F,
                de_CR=de_CR,
                # PSO parameters
                pso_pop_size=pso_pop_size,
                pso_inertia=pso_inertia,
                pso_cognitive=pso_cognitive,
                pso_social=pso_social,
                # GA parameters
                ga_pop_size=ga_pop_size,
                ga_crossover_rate=ga_crossover_rate,
                ga_mutation_rate=ga_mutation_rate,
                # Hybrid weights
                de_weight=de_weight,
                pso_weight=pso_weight,
                ga_weight=ga_weight
            )

            total_performance += performance

        return total_performance / num_runs

    def run_hybrid_meta_optimization(self,
                                     meta_pop_size: int = 15,
                                     meta_generations: int = 25,
                                     max_evaluations: int = 1200) -> Dict:
        """Run meta-PSO to optimize ALL hybrid algorithm parameters"""

        dim = len(self.hybrid_bounds)

        # Initialize meta-PSO particles
        particles = np.random.uniform(0, 1, (meta_pop_size, dim))
        velocities = np.random.uniform(-0.1, 0.1, (meta_pop_size, dim))

        def scale_particle(particle):
            scaled = []
            for i, (low, high) in enumerate(self.hybrid_bounds):
                scaled.append(low + particle[i] * (high - low))
            return scaled

        # Initialize personal and global bests
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(meta_pop_size, np.inf)

        logging.info("Evaluating initial hybrid parameter sets...")
        for i in range(meta_pop_size):
            scaled_params = scale_particle(particles[i])
            fitness = self.evaluate_hybrid_parameters(scaled_params, max_evaluations, num_runs=2)
            personal_best_scores[i] = fitness
            if i % 5 == 0:
                logging.info(f"  Particle {i}: fitness = {fitness:.6f}")

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = particles[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

        # PSO parameters for meta-optimization
        inertia = 0.7
        cognitive_weight = 1.5
        social_weight = 1.5

        convergence_history = [global_best_score]

        logging.info(f"Starting hybrid meta-optimization")
        logging.info(f"Initial best fitness: {global_best_score:.6f}")

        for generation in range(meta_generations):
            for i in range(meta_pop_size):
                # Update velocity and position
                r1, r2 = np.random.random(2)
                cognitive = cognitive_weight * r1 * (personal_best_positions[i] - particles[i])
                social = social_weight * r2 * (global_best_position - particles[i])
                velocities[i] = inertia * velocities[i] + cognitive + social
                particles[i] = np.clip(particles[i] + velocities[i], 0, 1)

                # Evaluate
                scaled_params = scale_particle(particles[i])
                fitness = self.evaluate_hybrid_parameters(scaled_params, max_evaluations, num_runs=1)

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

        # Get best parameters
        best_scaled_params = scale_particle(global_best_position)

        # Final evaluation with more runs
        final_performance = self.evaluate_hybrid_parameters(
            best_scaled_params, max_evaluations, num_runs=5
        )

        # Calculate final weights
        de_weight = best_scaled_params[10]
        pso_weight = best_scaled_params[11]
        ga_weight = 1.0 - de_weight - pso_weight

        result = {
            'best_parameters': best_scaled_params,
            'best_fitness': final_performance,
            'convergence_history': convergence_history,
            'algorithm_weights': {
                'DE': de_weight,
                'PSO': pso_weight,
                'GA': ga_weight
            }
        }

        return result


def main_hybrid():
    """Main function for hybrid algorithm optimization"""
    print("HYBRID ALGORITHM META-OPTIMIZATION")
    print("DE + PSO + GA COMBINED")
    print("=" * 60)

    # Create hybrid meta-optimizer
    meta_optimizer = HybridMetaPSO(test_function_name='rastrigin', dim=5)

    # Run hybrid meta-optimization
    result = meta_optimizer.run_hybrid_meta_optimization(
        meta_pop_size=12,
        meta_generations=20,
        max_evaluations=1000
    )

    print(f"\nHYBRID OPTIMIZATION RESULTS:")
    print(f"Best Overall Fitness: {result['best_fitness']:.6f}")

    # Display optimized parameters
    param_names = [
        'de_pop_size', 'de_F', 'de_CR',
        'pso_pop_size', 'pso_inertia', 'pso_cognitive', 'pso_social',
        'ga_pop_size', 'ga_crossover_rate', 'ga_mutation_rate',
        'de_weight', 'pso_weight'  # ga_weight is computed
    ]

    print(f"\nOptimized Hybrid Parameters:")
    for i, (name, value) in enumerate(zip(param_names, result['best_parameters'])):
        if 'pop_size' in name:
            value = int(value)
            print(f"  {name:20s}: {value:4d}")
        else:
            print(f"  {name:20s}: {value:.4f}")

    print(f"\nFinal Algorithm Weights:")
    for algo, weight in result['algorithm_weights'].items():
        print(f"  {algo:10s}: {weight:.3f} ({weight * 100:.1f}%)")

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(result['convergence_history'], linewidth=2)
    plt.xlabel('Meta-Generation')
    plt.ylabel('Best Fitness')
    plt.title('Hybrid Algorithm Meta-Optimization Convergence\n(DE + PSO + GA Combined)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    # Compare with individual algorithms
    print(f"\n{'=' * 50}")
    print("COMPARISON WITH INDIVIDUAL ALGORITHMS")
    print(f"{'=' * 50}")

    # Test individual algorithms with default parameters
    hybrid_final = result['best_fitness']

    # For fair comparison, give individual algorithms the same total budget
    individual_budget = 1000

    # Test individual algorithms
    individual_results = {}
    test_func = BenchmarkFunctions.get_function('rastrigin')
    dim = 5

    # DE only
    de_algo = HybridAlgorithm(test_func, dim)
    de_perf = de_algo.run_hybrid(
        max_evaluations=individual_budget,
        de_pop_size=30, de_F=0.5, de_CR=0.7,
        pso_pop_size=0,  # Disable PSO
        ga_pop_size=0,  # Disable GA
        de_weight=1.0, pso_weight=0.0, ga_weight=0.0
    )
    individual_results['DE_only'] = de_perf

    # PSO only
    pso_perf = de_algo.run_hybrid(
        max_evaluations=individual_budget,
        de_pop_size=0,  # Disable DE
        pso_pop_size=20, pso_inertia=0.7, pso_cognitive=1.5, pso_social=1.5,
        ga_pop_size=0,  # Disable GA
        de_weight=0.0, pso_weight=1.0, ga_weight=0.0
    )
    individual_results['PSO_only'] = pso_perf

    # GA only
    ga_perf = de_algo.run_hybrid(
        max_evaluations=individual_budget,
        de_pop_size=0,  # Disable DE
        pso_pop_size=0,  # Disable PSO
        ga_pop_size=25, ga_crossover_rate=0.8, ga_mutation_rate=0.1,
        de_weight=0.0, pso_weight=0.0, ga_weight=1.0
    )
    individual_results['GA_only'] = ga_perf



if __name__ == "__main__":
    # Run the hybrid optimization
    hybrid_result = main_hybrid()