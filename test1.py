"""
Differential Evolution with Beta Parameter Analysis
Project for Heuristics and Optimization Course
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import warnings
from typing import Dict, List, Tuple, Optional, Any
import json

warnings.filterwarnings('ignore')

class DEBetaProject:
    """
    Project for studying Differential Evolution with beta parameter variation
    """

    def __init__(self):
        self.results = {}
        self.analysis_data = {}

    # ==================== CHARACTERIZE FUNCTIONS ====================

    def characterize_function(self, x: np.ndarray, function_type: str, **params) -> float:
        """Characterize different optimization functions"""
        if function_type == 'sphere':
            return self._sphere_function(x)
        elif function_type == 'rastrigin':
            return self._rastrigin_function(x, **params)
        elif function_type == 'rosenbrock':
            return self._rosenbrock_function(x)
        elif function_type == 'ackley':
            return self._ackley_function(x, **params)
        elif function_type == 'griewank':
            return self._griewank_function(x)
        elif function_type == 'schwefel':
            return self._schwefel_function(x)
        else:
            raise ValueError(f"Unknown function type: {function_type}")

    def _sphere_function(self, x: np.ndarray) -> float:
        """Simple convex sphere function"""
        return np.sum(x**2)

    def _rastrigin_function(self, x: np.ndarray, A: float = 10.0) -> float:
        """Highly multimodal function with many local minima"""
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    def _rosenbrock_function(self, x: np.ndarray) -> float:
        """Valley-shaped function, hard to converge"""
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def _ackley_function(self, x: np.ndarray, a: float = 20, b: float = 0.2, c: float = 2*np.pi) -> float:
        """Many local minima, global minimum hard to find"""
        d = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)
        return term1 + term2 + a + np.exp(1)

    def _griewank_function(self, x: np.ndarray) -> float:
        """Many widespread local minima"""
        sum_part = np.sum(x**2) / 4000
        prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_part - prod_part + 1

    def _schwefel_function(self, x: np.ndarray) -> float:
        """Multimodal function with second-best minimum far from global optimum"""
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    # ==================== DIFFERENTIAL EVOLUTION WITH BETA VARIATION ====================

    def optimize_with_de_beta(self,
                             objective_func: callable,
                             bounds: List[Tuple[float, float]],
                             beta_values: List[float] = None,
                             de_strategy: str = 'best1bin',
                             popsize: int = 15,
                             recombination: float = 0.7,
                             maxiter: int = 1000) -> Dict[str, Any]:
        """Optimize using Differential Evolution with different beta (mutation) values"""

        if beta_values is None:
            beta_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

        results = {}

        for beta in beta_values:
            try:
                result = differential_evolution(
                    objective_func,
                    bounds,
                    strategy=de_strategy,
                    popsize=popsize,
                    mutation=beta,  # This is the beta parameter
                    recombination=recombination,
                    maxiter=maxiter,
                    disp=False,
                    polish=True
                )

                results[beta] = {
                    'success': result.success,
                    'optimal_value': result.fun,
                    'optimal_parameters': result.x,
                    'iterations': result.nit,
                    'function_evals': result.nfev,
                    'message': result.message,
                    'beta': beta
                }

            except Exception as e:
                print(f"Differential Evolution failed for beta={beta}: {e}")
                results[beta] = {
                    'success': False,
                    'optimal_value': None,
                    'optimal_parameters': None,
                    'iterations': 0,
                    'function_evals': 0,
                    'message': str(e),
                    'beta': beta
                }

        return results

    def optimize_with_de_strategy_beta(self,
                                      objective_func: callable,
                                      bounds: List[Tuple[float, float]],
                                      beta_values: List[float] = None,
                                      strategies: List[str] = None,
                                      popsize: int = 15,
                                      recombination: float = 0.7,
                                      maxiter: int = 1000) -> Dict[str, Any]:
        """Optimize using Differential Evolution with different strategies and beta values"""

        if beta_values is None:
            beta_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

        if strategies is None:
            strategies = ['best1bin', 'rand1bin', 'best2bin', 'rand2bin']

        results = {}

        for strategy in strategies:
            results[strategy] = {}
            print(f"Testing strategy: {strategy}")

            for beta in beta_values:
                try:
                    result = differential_evolution(
                        objective_func,
                        bounds,
                        strategy=strategy,
                        popsize=popsize,
                        mutation=beta,
                        recombination=recombination,
                        maxiter=maxiter,
                        disp=False,
                        polish=True
                    )

                    results[strategy][beta] = {
                        'success': result.success,
                        'optimal_value': result.fun,
                        'optimal_parameters': result.x,
                        'iterations': result.nit,
                        'function_evals': result.nfev,
                        'message': result.message,
                        'beta': beta,
                        'strategy': strategy
                    }

                except Exception as e:
                    print(f"DE failed for strategy={strategy}, beta={beta}: {e}")
                    results[strategy][beta] = {
                        'success': False,
                        'optimal_value': None,
                        'optimal_parameters': None,
                        'iterations': 0,
                        'function_evals': 0,
                        'message': str(e),
                        'beta': beta,
                        'strategy': strategy
                    }

        return results

    # ==================== VISUALIZATION METHODS ====================

    def plot_beta_analysis(self, results: Dict, title: str = "Beta Parameter Analysis"):
        """Create comprehensive visualization for beta parameter analysis"""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Extract data
        betas = []
        optimal_values = []
        function_evals = []
        success_flags = []
        iterations = []

        for beta, result_data in results.items():
            betas.append(beta)
            if result_data['success'] and result_data['optimal_value'] is not None:
                optimal_values.append(result_data['optimal_value'])
                function_evals.append(result_data['function_evals'])
                iterations.append(result_data['iterations'])
                success_flags.append(1)
            else:
                optimal_values.append(float('inf'))
                function_evals.append(0)
                iterations.append(0)
                success_flags.append(0)

        # 1. Optimal Values vs Beta
        successful_betas = [b for b, s in zip(betas, success_flags) if s == 1]
        successful_values = [v for v, s in zip(optimal_values, success_flags) if s == 1]

        if successful_betas:
            ax1.plot(successful_betas, successful_values, 'bo-', linewidth=2, markersize=8, label='Optimal Value')
            ax1.set_xlabel('Beta (Mutation) Value')
            ax1.set_ylabel('Optimal Function Value')
            ax1.set_title('Optimal Value vs Beta Parameter\n(Lower is Better)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Add value annotations
            for i, (beta, value) in enumerate(zip(successful_betas, successful_values)):
                ax1.annotate(f'{value:.4f}', (beta, value),
                            textcoords="offset points", xytext=(0,10), ha='center')

        # 2. Function Evaluations vs Beta
        successful_evals = [e for e, s in zip(function_evals, success_flags) if s == 1]
        if successful_evals:
            ax2.plot(successful_betas, successful_evals, 'ro-', linewidth=2, markersize=8, label='Function Evaluations')
            ax2.set_xlabel('Beta (Mutation) Value')
            ax2.set_ylabel('Number of Function Evaluations')
            ax2.set_title('Computational Cost vs Beta Parameter\n(Lower is Better)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # Add value annotations
            for i, (beta, evals) in enumerate(zip(successful_betas, successful_evals)):
                ax2.annotate(f'{evals}', (beta, evals),
                            textcoords="offset points", xytext=(0,10), ha='center')

        # 3. Success Rate
        success_rate = np.mean(success_flags) * 100
        ax3.bar(['Success Rate'], [success_rate], color='green' if success_rate > 50 else 'red', alpha=0.7)
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_title(f'Overall Success Rate: {success_rate:.1f}%')
        ax3.set_ylim(0, 100)
        ax3.text(0, success_rate + 2, f'{success_rate:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 4. Best Beta Performance
        if successful_betas:
            best_idx = np.argmin(successful_values)
            best_beta = successful_betas[best_idx]
            best_value = successful_values[best_idx]
            best_evals = successful_evals[best_idx]

            metrics = ['Best Beta', 'Best Value', 'Evaluations']
            values = [best_beta, best_value, best_evals]
            colors = ['blue', 'green', 'orange']

            bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
            ax4.set_title('Best Performing Beta Parameter')
            ax4.set_ylabel('Value')

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.05,
                        f'{value:.4f}' if value != best_evals else f'{value:.0f}',
                        ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_strategy_beta_comparison(self, results: Dict, title: str = "Strategy vs Beta Analysis"):
        """Compare different strategies across beta values"""

        strategies = list(results.keys())
        beta_values = list(next(iter(results.values())).keys())

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Prepare data
        strategy_data = {}
        for strategy in strategies:
            strategy_data[strategy] = {
                'betas': [],
                'values': [],
                'evals': [],
                'success_rate': 0
            }

            success_count = 0
            for beta in beta_values:
                result = results[strategy][beta]
                strategy_data[strategy]['betas'].append(beta)
                if result['success'] and result['optimal_value'] is not None:
                    strategy_data[strategy]['values'].append(result['optimal_value'])
                    strategy_data[strategy]['evals'].append(result['function_evals'])
                    success_count += 1
                else:
                    strategy_data[strategy]['values'].append(float('inf'))
                    strategy_data[strategy]['evals'].append(0)

            strategy_data[strategy]['success_rate'] = success_count / len(beta_values) * 100

        # 1. Optimal Values Heatmap
        value_matrix = np.zeros((len(strategies), len(beta_values)))
        for i, strategy in enumerate(strategies):
            for j, beta in enumerate(beta_values):
                value = results[strategy][beta]['optimal_value']
                value_matrix[i, j] = value if value is not None else float('inf')

        im1 = ax1.imshow(value_matrix, cmap='viridis', aspect='auto')
        ax1.set_xticks(range(len(beta_values)))
        ax1.set_xticklabels([f'{b:.1f}' for b in beta_values])
        ax1.set_yticks(range(len(strategies)))
        ax1.set_yticklabels(strategies)
        ax1.set_xlabel('Beta Value')
        ax1.set_ylabel('Strategy')
        ax1.set_title('Optimal Values Heatmap\n(Darker = Better)')

        # Add value annotations
        for i in range(len(strategies)):
            for j in range(len(beta_values)):
                if value_matrix[i, j] != float('inf'):
                    ax1.text(j, i, f'{value_matrix[i, j]:.2f}',
                            ha="center", va="center", color="white", fontsize=8)

        plt.colorbar(im1, ax=ax1)

        # 2. Success Rates by Strategy
        success_rates = [strategy_data[s]['success_rate'] for s in strategies]
        bars = ax2.bar(strategies, success_rates, color='lightblue', alpha=0.7)
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate by DE Strategy')
        ax2.set_ylim(0, 100)

        for bar, rate in zip(bars, success_rates):
            ax2.text(bar.get_x() + bar.get_width()/2., rate + 2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 3. Best Beta for Each Strategy
        best_betas = []
        best_values = []
        for strategy in strategies:
            valid_values = [(beta, val) for beta, val in zip(strategy_data[strategy]['betas'],
                                                           strategy_data[strategy]['values'])
                          if val != float('inf')]
            if valid_values:
                best_beta, best_val = min(valid_values, key=lambda x: x[1])
                best_betas.append(best_beta)
                best_values.append(best_val)
            else:
                best_betas.append(0)
                best_values.append(float('inf'))

        x_pos = np.arange(len(strategies))
        width = 0.35

        ax3.bar(x_pos - width/2, best_betas, width, label='Best Beta', color='orange', alpha=0.7)
        ax3.bar(x_pos + width/2, best_values, width, label='Best Value', color='green', alpha=0.7)
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Value')
        ax3.set_title('Best Beta and Corresponding Value by Strategy')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(strategies)
        ax3.legend()

        # 4. Computational Efficiency
        avg_evals = []
        for strategy in strategies:
            valid_evals = [e for e in strategy_data[strategy]['evals'] if e > 0]
            avg_evals.append(np.mean(valid_evals) if valid_evals else 0)

        ax4.bar(strategies, avg_evals, color='red', alpha=0.7)
        ax4.set_ylabel('Average Function Evaluations')
        ax4.set_title('Computational Cost by Strategy\n(Lower is Better)')

        for i, evals in enumerate(avg_evals):
            if evals > 0:
                ax4.text(i, evals + max(avg_evals)*0.02, f'{evals:.0f}',
                        ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        return fig

    # ==================== UTILITY FUNCTIONS ====================

    def save_results(self, filename: str = 'de_beta_results.json'):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=self._json_serializer)
        print(f"Results saved to {filename}")

    def load_results(self, filename: str = 'de_beta_results.json'):
        """Load results from JSON file"""
        with open(filename, 'r') as f:
            self.results = json.load(f)
        print(f"Results loaded from {filename}")

    def _json_serializer(self, obj):
        """Helper function to serialize numpy types for JSON"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return str(obj)

# ==================== QUICK DEMONSTRATION ====================

def quick_demo():
    """Quick demonstration of Differential Evolution with beta parameter variation"""

    print("DIFFERENTIAL EVOLUTION BETA PARAMETER ANALYSIS")
    print("=" * 60)

    project = DEBetaProject()

    # Test configurations
    functions = ['rastrigin', 'ackley', 'sphere']
    bounds = [(-5, 5)] * 2  # 2D problem
    beta_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    all_results = {}

    for func_name in functions:
        print(f"\n{'='*50}")
        print(f"Testing {func_name.upper()} Function")
        print(f"{'='*50}")

        # Define objective function
        def objective(x):
            return project.characterize_function(x, func_name)

        # Test different beta values
        print("Testing different beta values with best1bin strategy...")
        beta_results = project.optimize_with_de_beta(
            objective, bounds, beta_values=beta_values, de_strategy='best1bin', maxiter=500
        )

        all_results[func_name] = beta_results

        # Print results
        print(f"\nResults for {func_name}:")
        print("Beta\tSuccess\tOptimal Value\tEvaluations")
        print("-" * 50)
        for beta, result in beta_results.items():
            if result['success']:
                print(f"{beta:.1f}\t✓\t{result['optimal_value']:.6f}\t{result['function_evals']}")
            else:
                print(f"{beta:.1f}\t✗\tFailed\t\t-")

        # Create visualization for this function
        fig = project.plot_beta_analysis(beta_results, f"Beta Analysis - {func_name.title()} Function")
        plt.savefig(f'beta_analysis_{func_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Compare strategies across beta values for the most challenging function
    print(f"\n{'='*50}")
    print("COMPARING STRATEGIES ACROSS BETA VALUES")
    print(f"{'='*50}")

    def rastrigin_obj(x):
        return project.characterize_function(x, 'rastrigin')

    strategies = ['best1bin', 'rand1bin', 'best2bin', 'rand2bin']
    strategy_results = project.optimize_with_de_strategy_beta(
        rastrigin_obj, bounds, beta_values=beta_values, strategies=strategies, maxiter=500
    )

    all_results['strategy_comparison'] = strategy_results

    # Print strategy comparison
    print("\nStrategy Comparison for Rastrigin Function:")
    print("Strategy\tBest Beta\tBest Value\tSuccess Rate")
    print("-" * 60)
    for strategy in strategies:
        valid_results = [r for r in strategy_results[strategy].values() if r['success']]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['optimal_value'])
            success_rate = len(valid_results) / len(beta_values) * 100
            print(f"{strategy}\t{best_result['beta']:.1f}\t\t{best_result['optimal_value']:.6f}\t{success_rate:.1f}%")
        else:
            print(f"{strategy}\t-\t\t-\t\t0.0%")

    # Create strategy comparison visualization
    fig = project.plot_strategy_beta_comparison(strategy_results, "DE Strategy vs Beta Analysis - Rastrigin Function")
    plt.savefig('strategy_beta_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save all results
    project.results = all_results
    project.save_results('quick_demo_beta_results.json')

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")

    # Overall analysis
    for func_name in functions:
        beta_results = all_results[func_name]
        success_count = sum(1 for r in beta_results.values() if r['success'])
        best_result = min([r for r in beta_results.values() if r['success']],
                         key=lambda x: x['optimal_value'], default=None)

        if best_result:
            print(f"{func_name.upper()}: {success_count}/{len(beta_values)} successful, "
                  f"Best: beta={best_result['beta']}, value={best_result['optimal_value']:.6f}")

    print(f"\nVisualizations saved as:")
    for func_name in functions:
        print(f"  - beta_analysis_{func_name}.png")
    print("  - strategy_beta_comparison.png")
    print("  - quick_demo_beta_results.json")

if __name__ == "__main__":
    quick_demo()