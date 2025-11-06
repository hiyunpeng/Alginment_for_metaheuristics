import numpy as np
import math
from typing import List, Tuple, Optional

# Constants
INF = 1.0e99
EPS = 1.0e-14
E = 2.7182818284590452353602874713526625
PI = 3.1415926535897932384626433832795029


class CEC13:
    def __init__(self, nx: int):
        self.nx = nx
        self.cf_num = 10

        # Load M and OShift data
        self.M = self._load_M(nx)
        self.OShift = self._load_OShift()

    def _load_M(self, nx: int) -> np.ndarray:
        """Load M matrix from file"""
        try:
            filename = f"./inst/extdata/M_D{nx}.txt"
            data = np.loadtxt(filename)
            return data.reshape(self.cf_num, nx, nx)
        except:
            print(f"Warning: Could not load M matrix file for nx={nx}")
            # Return identity matrices as fallback
            return np.array([np.eye(nx) for _ in range(self.cf_num)])

    def _load_OShift(self) -> np.ndarray:
        """Load OShift vector from file"""
        try:
            filename = "./inst/extdata/shift_data.txt"
            data = np.loadtxt(filename)
            return data.reshape(self.cf_num, self.nx)
        except FileNotFoundError:
            print(f"Error: File not found at {filename}")
        except ValueError as e:
            print(f"Error: Could not parse the file {filename}. Ensure it contains valid numeric data. Details: {e}")
        except Exception as e:
            print(f"Unexpected error while loading OShift: {e}")
        # Return zeros as fallback
        return np.zeros((self.cf_num, self.nx))

    def test_func(self, x: np.ndarray, func_num: int) -> float:
        """Main test function interface"""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        mx, nx = x.shape
        f = np.zeros(mx)

        for i in range(mx):
            if func_num == 1:
                f[i] = self.sphere_func(x[i], 0)
            elif func_num == 2:
                f[i] = self.ellips_func(x[i], 1)
            elif func_num == 3:
                f[i] = self.bent_cigar_func(x[i], 1)
            elif func_num == 4:
                f[i] = self.discus_func(x[i], 1)
            elif func_num == 5:
                f[i] = self.dif_powers_func(x[i], 0)
            elif func_num == 6:
                f[i] = self.rosenbrock_func(x[i], 1)
            elif func_num == 7:
                f[i] = self.schaffer_F7_func(x[i], 1)
            elif func_num == 8:
                f[i] = self.ackley_func(x[i], 1)
            elif func_num == 9:
                f[i] = self.weierstrass_func(x[i], 1)
            elif func_num == 10:
                f[i] = self.griewank_func(x[i], 1)
            elif func_num == 11:
                f[i] = self.rastrigin_func(x[i], 0)
            elif func_num == 12:
                f[i] = self.rastrigin_func(x[i], 1)
            elif func_num == 13:
                f[i] = self.step_rastrigin_func(x[i], 1)
            elif func_num == 14:
                f[i] = self.schwefel_func(x[i], 0)
            elif func_num == 15:
                f[i] = self.schwefel_func(x[i], 1)
            elif func_num == 16:
                f[i] = self.katsuura_func(x[i], 1)
            elif func_num == 17:
                f[i] = self.bi_rastrigin_func(x[i], 0)
            elif func_num == 18:
                f[i] = self.bi_rastrigin_func(x[i], 1)
            elif func_num == 19:
                f[i] = self.grie_rosen_func(x[i], 1)
            elif func_num == 20:
                f[i] = self.escaffer6_func(x[i], 1)
            elif func_num == 21:
                f[i] = self.cf01(x[i], 1)
            elif func_num == 22:
                f[i] = self.cf02(x[i], 0)
            elif func_num == 23:
                f[i] = self.cf03(x[i], 1)
            elif func_num == 24:
                f[i] = self.cf04(x[i], 1)
            elif func_num == 25:
                f[i] = self.cf05(x[i], 1)
            elif func_num == 26:
                f[i] = self.cf06(x[i], 1)
            elif func_num == 27:
                f[i] = self.cf07(x[i], 1)
            elif func_num == 28:
                f[i] = self.cf08(x[i], 1)
            else:
                raise ValueError(f"Invalid function number: {func_num}")

        return f[0] if mx == 1 else f

    def shiftfunc(self, x: np.ndarray, Os: np.ndarray) -> np.ndarray:
        """Shift function"""
        return x - Os

    def rotatefunc(self, x: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Rotate function"""
        return x @ M.T

    def asyfunc(self, x: np.ndarray, beta: float) -> np.ndarray:
        """Asymmetric function"""
        nx = len(x)
        xasy = x.copy()
        for i in range(nx):
            if x[i] > 0:
                xasy[i] = x[i] ** (1.0 + beta * i / (nx - 1) * math.sqrt(x[i]))
        return xasy

    def oszfunc(self, x: np.ndarray) -> np.ndarray:
        """Osz function"""
        nx = len(x)
        xosz = x.copy()

        for i in [0, nx - 1]:
            if x[i] != 0:
                xx = math.log(abs(x[i]))
                if x[i] > 0:
                    c1, c2 = 10.0, 7.9
                else:
                    c1, c2 = 5.5, 3.1

                sx = 1 if x[i] > 0 else (-1 if x[i] < 0 else 0)
                xosz[i] = sx * math.exp(xx + 0.049 * (math.sin(c1 * xx) + math.sin(c2 * xx)))

        return xosz

    def sphere_func(self, x: np.ndarray, r_flag: int) -> float:
        """Sphere function"""
        y = self.shiftfunc(x, self.OShift[0])
        if r_flag == 1:
            z = self.rotatefunc(y, self.M[0])
        else:
            z = y.copy()
        return np.sum(z ** 2)

    def ellips_func(self, x: np.ndarray, r_flag: int) -> float:
        """Ellipsoidal function"""
        y = self.shiftfunc(x, self.OShift[1])
        if r_flag == 1:
            z = self.rotatefunc(y, self.M[1])
        else:
            z = y.copy()
        y_osz = self.oszfunc(z)

        result = 0.0
        for i in range(self.nx):
            result += (10.0 ** (6.0 * i / (self.nx - 1))) * y_osz[i] ** 2
        return result

    def bent_cigar_func(self, x: np.ndarray, r_flag: int) -> float:
        """Bent Cigar function"""
        beta = 0.5
        y = self.shiftfunc(x, self.OShift[2])
        if r_flag == 1:
            z = self.rotatefunc(y, self.M[2])
        else:
            z = y.copy()

        y_asy = self.asyfunc(z, beta)
        if r_flag == 1:
            z = self.rotatefunc(y_asy, self.M[2])
        else:
            z = y_asy.copy()

        result = z[0] ** 2
        for i in range(1, self.nx):
            result += (10.0 ** 6.0) * z[i] ** 2
        return result

    def discus_func(self, x: np.ndarray, r_flag: int) -> float:
        """Discus function"""
        y = self.shiftfunc(x, self.OShift[3])
        if r_flag == 1:
            z = self.rotatefunc(y, self.M[3])
        else:
            z = y.copy()
        y_osz = self.oszfunc(z)

        result = (10.0 ** 6.0) * y_osz[0] ** 2
        for i in range(1, self.nx):
            result += y_osz[i] ** 2
        return result

    def dif_powers_func(self, x: np.ndarray, r_flag: int) -> float:
        """Different Powers function"""
        y = self.shiftfunc(x, self.OShift[4])
        if r_flag == 1:
            z = self.rotatefunc(y, self.M[4])
        else:
            z = y.copy()

        result = 0.0
        for i in range(self.nx):
            result += abs(z[i]) ** (2 + 4 * i / (self.nx - 1))
        return result ** 0.5

    def rosenbrock_func(self, x: np.ndarray, r_flag: int) -> float:
        """Rosenbrock's function"""
        y = self.shiftfunc(x, self.OShift[5])
        y = y * 2.048 / 100.0  # shrink to original search range

        if r_flag == 1:
            z = self.rotatefunc(y, self.M[5])
        else:
            z = y.copy()

        z = z + 1  # shift to origin

        result = 0.0
        for i in range(self.nx - 1):
            tmp1 = z[i] ** 2 - z[i + 1]
            tmp2 = z[i] - 1.0
            result += 100.0 * tmp1 ** 2 + tmp2 ** 2
        return result

    def schaffer_F7_func(self, x: np.ndarray, r_flag: int) -> float:
        """Schaffer's F7 function"""
        y = self.shiftfunc(x, self.OShift[6])
        if r_flag == 1:
            z = self.rotatefunc(y, self.M[6])
        else:
            z = y.copy()

        y_asy = self.asyfunc(z, 0.5)
        z = y_asy * (10.0 ** (np.arange(self.nx) / (self.nx - 1) / 2.0))

        if r_flag == 1:
            y = self.rotatefunc(z, self.M[6])
        else:
            y = z.copy()

        z_vals = np.sqrt(y[:-1] ** 2 + y[1:] ** 2)

        result = 0.0
        for i in range(len(z_vals)):
            tmp = math.sin(50.0 * z_vals[i] ** 0.2)
            result += z_vals[i] ** 0.5 + z_vals[i] ** 0.5 * tmp ** 2

        return (result ** 2) / ((self.nx - 1) ** 2)

    def ackley_func(self, x: np.ndarray, r_flag: int) -> float:
        """Ackley's function"""
        y = self.shiftfunc(x, self.OShift[7])
        if r_flag == 1:
            z = self.rotatefunc(y, self.M[7])
        else:
            z = y.copy()

        y_asy = self.asyfunc(z, 0.5)
        z = y_asy * (10.0 ** (np.arange(self.nx) / (self.nx - 1) / 2.0))

        if r_flag == 1:
            y = self.rotatefunc(z, self.M[7])
        else:
            y = z.copy()

        sum1 = np.sum(y ** 2)
        sum2 = np.sum(np.cos(2.0 * PI * y))

        sum1 = -0.2 * math.sqrt(sum1 / self.nx)
        sum2 /= self.nx

        return E - 20.0 * math.exp(sum1) - math.exp(sum2) + 20.0

    def weierstrass_func(self, x: np.ndarray, r_flag: int) -> float:
        """Weierstrass's function"""
        y = self.shiftfunc(x, self.OShift[8])
        y = y * 0.5 / 100.0  # shrink to original search range

        if r_flag == 1:
            z = self.rotatefunc(y, self.M[8])
        else:
            z = y.copy()

        y_asy = self.asyfunc(z, 0.5)
        z = y_asy * (10.0 ** (np.arange(self.nx) / (self.nx - 1) / 2.0))

        if r_flag == 1:
            y = self.rotatefunc(z, self.M[8])
        else:
            y = z.copy()

        a, b, k_max = 0.5, 3.0, 20
        result = 0.0
        sum2 = 0.0

        # Precompute sum2
        for j in range(k_max + 1):
            sum2 += (a ** j) * math.cos(2.0 * PI * (b ** j) * 0.5)

        for i in range(self.nx):
            sum_val = 0.0
            for j in range(k_max + 1):
                sum_val += (a ** j) * math.cos(2.0 * PI * (b ** j) * (y[i] + 0.5))
            result += sum_val

        return result - self.nx * sum2

    def griewank_func(self, x: np.ndarray, r_flag: int) -> float:
        """Griewank's function"""
        y = self.shiftfunc(x, self.OShift[9])
        y = y * 600.0 / 100.0  # shrink to original search range

        if r_flag == 1:
            z = self.rotatefunc(y, self.M[9])
        else:
            z = y.copy()

        z = z * (100.0 ** (np.arange(self.nx) / (self.nx - 1) / 2.0))

        s = np.sum(z ** 2)
        p = 1.0
        for i in range(self.nx):
            p *= math.cos(z[i] / math.sqrt(1.0 + i))

        return 1.0 + s / 4000.0 - p

    def rastrigin_func(self, x: np.ndarray, r_flag: int) -> float:
        """Rastrigin's function"""
        alpha, beta = 10.0, 0.2
        func_idx = 10 % self.cf_num
        y = self.shiftfunc(x, self.OShift[func_idx])
        y = y * 5.12 / 100.0  # shrink to original search range

        if r_flag == 1:
            z = self.rotatefunc(y, self.M[func_idx])
        else:
            z = y.copy()

        y_osz = self.oszfunc(z)
        z_asy = self.asyfunc(y_osz, beta)

        if r_flag == 1:
            y = self.rotatefunc(z_asy, self.M[func_idx])
        else:
            y = z_asy.copy()

        y = y * (alpha ** (np.arange(self.nx) / (self.nx - 1) / 2.0))

        if r_flag == 1:
            z = self.rotatefunc(y, self.M[func_idx])
        else:
            z = y.copy()

        result = 0.0
        for i in range(self.nx):
            result += z[i] ** 2 - 10.0 * math.cos(2.0 * PI * z[i]) + 10.0
        return result

    # Placeholder for other functions (implement as needed)
    def step_rastrigin_func(self, x: np.ndarray, r_flag: int) -> float:
        """Step Rastrigin function - placeholder"""
        return self.rastrigin_func(x, r_flag)

    def schwefel_func(self, x: np.ndarray, r_flag: int) -> float:
        """Schwefel function - placeholder"""
        return np.sum(x ** 2)

    def katsuura_func(self, x: np.ndarray, r_flag: int) -> float:
        """Katsuura function - placeholder"""
        return np.sum(x ** 2)

    def bi_rastrigin_func(self, x: np.ndarray, r_flag: int) -> float:
        """Bi-Rastrigin function - placeholder"""
        return self.rastrigin_func(x, r_flag)

    def grie_rosen_func(self, x: np.ndarray, r_flag: int) -> float:
        """Griewank-Rosenbrock function - placeholder"""
        return self.rosenbrock_func(x, r_flag)

    def escaffer6_func(self, x: np.ndarray, r_flag: int) -> float:
        """Escaffer6 function - placeholder"""
        return np.sum(x ** 2)

    def cf01(self, x: np.ndarray, r_flag: int) -> float:
        """Composition Function 1 - placeholder"""
        return self.sphere_func(x, r_flag)

    def cf02(self, x: np.ndarray, r_flag: int) -> float:
        """Composition Function 2 - placeholder"""
        return self.sphere_func(x, r_flag)

    def cf03(self, x: np.ndarray, r_flag: int) -> float:
        """Composition Function 3 - placeholder"""
        return self.sphere_func(x, r_flag)

    def cf04(self, x: np.ndarray, r_flag: int) -> float:
        """Composition Function 4 - placeholder"""
        return self.sphere_func(x, r_flag)

    def cf05(self, x: np.ndarray, r_flag: int) -> float:
        """Composition Function 5 - placeholder"""
        return self.sphere_func(x, r_flag)

    def cf06(self, x: np.ndarray, r_flag: int) -> float:
        """Composition Function 6 - placeholder"""
        return self.sphere_func(x, r_flag)

    def cf07(self, x: np.ndarray, r_flag: int) -> float:
        """Composition Function 7 - placeholder"""
        return self.sphere_func(x, r_flag)

    def cf08(self, x: np.ndarray, r_flag: int) -> float:
        """Composition Function 8 - placeholder"""
        return self.sphere_func(x, r_flag)


# PSO implementation
class PSO:
    def __init__(self, nx: int, np: int, T_max: int):
        self.nx = nx
        self.np = np
        self.T_max = T_max
        self.cec = CEC13(nx)

    def optimize(self, func_num: int, w: float, a: float) -> float:
        """PSO optimization for a given function"""
        a1 = 0.5 * a
        a2 = a - a1

        # Initialize particles
        x = np.random.uniform(-0.5, 0.5, (self.np, self.nx)) * 200.0
        v = np.random.uniform(-1.0, 1.0, (self.np, self.nx)) * 2.0

        px = x.copy()  # personal best positions
        pfx = np.full(self.np, np.inf)  # personal best fitness

        gx = np.random.uniform(-0.5, 0.5, self.nx) * 200.0  # global best position
        gfx = np.inf  # global best fitness

        for t in range(self.T_max):
            # Evaluate fitness
            fx = np.array([self.cec.test_func(x[i], func_num) for i in range(self.np)])

            # Update personal best
            for n in range(self.np):
                if fx[n] < pfx[n]:
                    pfx[n] = fx[n]
                    px[n] = x[n].copy()

            # Update global best
            best_idx = np.argmin(fx)
            if fx[best_idx] < gfx:
                gfx = fx[best_idx]
                gx = x[best_idx].copy()

            # Update velocity and position
            for i in range(self.np):
                for j in range(self.nx):
                    r1, r2 = np.random.random(), np.random.random()
                    v[i, j] = (w * v[i, j] +
                               a1 * r1 * (px[i, j] - x[i, j]) +
                               a2 * r2 * (gx[j] - x[i, j]))
                    x[i, j] += v[i, j]

            # Check for overflow
            if np.any(np.abs(x) > 1e6) or np.any(np.abs(v) > 1e6):
                break

        return gfx

def main():
    """Main function to run the benchmark"""
    T_max = 500  # PSO run time
    rr = 10  # statistics: trials
    nx = 5  # dimension
    np = 20  # number of particles

    pso = PSO(nx, np, T_max)

    # Test all functions
    results = []
    for func_num in range(1, 29):
        best_fitness = float('inf')  # Initialize best fitness
        best_params = None  # Initialize best parameters
        for w in [0.4, 0.5, 0.6]:  # Example range of inertia weights
            for a in [1.2, 1.5, 1.8]:  # Example range of acceleration coefficients
                func_results = []  # Initialize a list to store results for each trial
                for _ in range(rr):
                    fitness = pso.optimize(func_num, w=w, a=a)  # Optimize with given parameters
                    func_results.append(fitness)  # Append fitness to the list
                avg_fitness = np.mean(func_results)  # Compute the mean of the results
                if avg_fitness < best_fitness:  # Update best fitness and parameters
                    best_fitness = avg_fitness
                    best_params = (w, a)
        results.append((func_num, best_fitness, best_params))  # Store results for the function

    return results

if __name__ == "__main__":
    results = main()
