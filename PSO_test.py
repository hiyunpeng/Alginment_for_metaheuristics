import math
import random

INF = 1.0e99
EPS = 1.0e-14
E = 2.7182818284590452353602874713526625
PI = 3.1415926535897932384626433832795029


def shiftfunc(x, Os):
    nx = len(x)
    return [x[i] - Os[i] for i in range(nx)]


# REMOVED: rotatefunc since we're not using rotation matrices anymore

def asyfunc(x, beta):
    """Asymmetric transformation."""
    nx = len(x)
    y = x[:]  # start from original values (matches C behaviour)
    if nx <= 1:
        return y

    MAX_VAL = 1e150  # large but safe; avoids inf in later math.floor()

    for i in range(nx):
        if y[i] > 0.0:
            exponent = 1.0 + beta * i / (nx - 1) * math.sqrt(y[i])
            try:
                val = math.pow(y[i], exponent)
            except OverflowError:
                val = MAX_VAL
            if val > MAX_VAL:
                val = MAX_VAL
            y[i] = val

    return y


def oszfunc(x):
    """Nonlinear oscillation transformation."""
    nx = len(x)
    xosz = x[:]  # default copy
    for i in range(nx):
        if i == 0 or i == nx - 1:
            if x[i] == 0.0:
                xosz[i] = 0.0
            else:
                xx = math.log(abs(x[i]))
                if x[i] > 0:
                    c1, c2 = 10.0, 7.9
                    sx = 1.0
                else:
                    c1, c2 = 5.5, 3.1
                    sx = -1.0
                xosz[i] = sx * math.exp(xx + 0.049 * (math.sin(c1 * xx) + math.sin(c2 * xx)))
        else:
            xosz[i] = x[i]
    return xosz


def cf_cal(x, nx, Os, delta, bias, fit, cf_num):
    w = [0.0] * cf_num
    w_max = 0.0
    for i in range(cf_num):
        fit[i] += bias[i]
        wi = 0.0
        base = i * nx
        for j in range(nx):
            diff = x[j] - Os[base + j]
            wi += diff * diff
        if wi != 0.0:
            wi = (1.0 / math.sqrt(wi)) * math.exp(-wi / (2.0 * nx * (delta[i] ** 2.0)))
        else:
            wi = INF
        w[i] = wi
        if wi > w_max:
            w_max = wi

    w_sum = sum(w)
    if w_max == 0.0:
        for i in range(cf_num):
            w[i] = 1.0
        w_sum = float(cf_num)

    f_val = 0.0
    for i in range(cf_num):
        f_val += (w[i] / w_sum) * fit[i]
    return f_val


# --- basic functions (modified to remove rotation) ---


def sphere_func(x, nx, Os):
    y = shiftfunc(x, Os)
    return sum(v * v for v in y)


def ellips_func(x, nx, Os):
    y = shiftfunc(x, Os)
    y2 = oszfunc(y)
    f = 0.0
    if nx == 1:
        return y2[0] * y2[0]
    for i in range(nx):
        f += math.pow(10.0, 6.0 * i / (nx - 1)) * y2[i] * y2[i]
    return f


def bent_cigar_func(x, nx, Os):
    beta = 0.5
    y = shiftfunc(x, Os)
    y2 = asyfunc(y, beta)
    f = y2[0] * y2[0]
    for i in range(1, nx):
        f += math.pow(10.0, 6.0) * y2[i] * y2[i]
    return f


def discus_func(x, nx, Os):
    y = shiftfunc(x, Os)
    y2 = oszfunc(y)
    f = math.pow(10.0, 6.0) * y2[0] * y2[0]
    for i in range(1, nx):
        f += y2[i] * y2[i]
    return f


def dif_powers_func(x, nx, Os):
    y = shiftfunc(x, Os)
    f = 0.0
    if nx == 1:
        return abs(y[0]) ** 2.0
    for i in range(nx):
        exponent = 2.0 + 4.0 * i / (nx - 1)
        f += math.pow(abs(y[i]), exponent)
    return math.sqrt(f)


def rosenbrock_func(x, nx, Os):
    y = shiftfunc(x, Os)
    y = [v * 2.048 / 100.0 for v in y]
    z = [v + 1.0 for v in y]
    f = 0.0
    for i in range(nx - 1):
        tmp1 = z[i] * z[i] - z[i + 1]
        tmp2 = z[i] - 1.0
        f += 100.0 * tmp1 * tmp1 + tmp2 * tmp2
    return f


def schaffer_F7_func(x, nx, Os):
    y = shiftfunc(x, Os)
    y2 = asyfunc(y, 0.5)
    z2 = [y2[i] * math.pow(10.0, 1.0 * i / (nx - 1) / 2.0) for i in range(nx)]
    if nx < 2:
        return 0.0
    z3 = [math.sqrt(z2[i] * z2[i] + z2[i + 1] * z2[i + 1]) for i in range(nx - 1)]
    f = 0.0
    for zi in z3:
        tmp = math.sin(50.0 * math.pow(zi, 0.2))
        f += math.pow(zi, 0.5) + math.pow(zi, 0.5) * tmp * tmp
    f = f * f / ((nx - 1) * (nx - 1))
    return f


def ackley_func(x, nx, Os):
    y = shiftfunc(x, Os)
    y2 = asyfunc(y, 0.5)
    z2 = [y2[i] * math.pow(10.0, 1.0 * i / (nx - 1) / 2.0) for i in range(nx)]

    sum1 = 0.0
    sum2 = 0.0
    for i in range(nx):
        sum1 += z2[i] * z2[i]
        sum2 += math.cos(2.0 * PI * z2[i])
    sum1 = -0.2 * math.sqrt(sum1 / nx)
    sum2 /= nx
    return E - 20.0 * math.exp(sum1) - math.exp(sum2) + 20.0


def weierstrass_func(x, nx, Os):
    y = shiftfunc(x, Os)
    y = [v * 0.5 / 100.0 for v in y]
    y2 = asyfunc(y, 0.5)
    z2 = [y2[i] * math.pow(10.0, 1.0 * i / (nx - 1) / 2.0) for i in range(nx)]

    a = 0.5
    b = 3.0
    k_max = 20
    f = 0.0
    sum2 = 0.0
    for i in range(nx):
        s = 0.0
        sum2 = 0.0
        for j in range(k_max + 1):
            a_pow = a ** j
            b_pow = b ** j
            s += a_pow * math.cos(2.0 * PI * b_pow * (z2[i] + 0.5))
            sum2 += a_pow * math.cos(2.0 * PI * b_pow * 0.5)
        f += s
    f -= nx * sum2
    return f


def griewank_func(x, nx, Os):
    y = shiftfunc(x, Os)
    y = [v * 600.0 / 100.0 for v in y]
    z2 = [y[i] * math.pow(100.0, 1.0 * i / (nx - 1) / 2.0) for i in range(nx)]

    s = 0.0
    p = 1.0
    for i in range(nx):
        s += z2[i] * z2[i]
        p *= math.cos(z2[i] / math.sqrt(1.0 + i))
    return 1.0 + s / 4000.0 - p


def rastrigin_func(x, nx, Os):
    alpha = 10.0
    beta = 0.2
    y = shiftfunc(x, Os)
    y = [v * 5.12 / 100.0 for v in y]
    y2 = oszfunc(y)
    z2 = asyfunc(y2, beta)
    for i in range(nx):
        z2[i] *= math.pow(alpha, 1.0 * i / (nx - 1) / 2.0)
    f = 0.0
    for i in range(nx):
        f += z2[i] * z2[i] - 10.0 * math.cos(2.0 * PI * z2[i]) + 10.0
    return f


def step_rastrigin_func(x, nx, Os):
    alpha = 10.0
    beta = 0.2
    y = shiftfunc(x, Os)
    y = [v * 5.12 / 100.0 for v in y]
    for i in range(nx):
        if abs(y[i]) > 0.5:
            y[i] = math.floor(2.0 * y[i] + 0.5) / 2.0
    y2 = oszfunc(y)
    z2 = asyfunc(y2, beta)
    for i in range(nx):
        z2[i] *= math.pow(alpha, 1.0 * i / (nx - 1) / 2.0)
    f = 0.0
    for i in range(nx):
        f += z2[i] * z2[i] - 10.0 * math.cos(2.0 * PI * z2[i]) + 10.0
    return f


def schwefel_func(x, nx, Os):
    y = shiftfunc(x, Os)
    y = [v * 1000.0 / 100.0 for v in y]
    z2 = [y[i] * math.pow(10.0, 1.0 * i / (nx - 1) / 2.0) for i in range(nx)]
    z2 = [z2[i] + 4.209687462275036e+002 for i in range(nx)]
    f = 0.0
    for i in range(nx):
        zi = z2[i]
        if zi > 500:
            f -= (500.0 - math.fmod(zi, 500.0)) * math.sin(math.sqrt(500.0 - math.fmod(zi, 500.0)))
            tmp = (zi - 500.0) / 100.0
            f += tmp * tmp / nx
        elif zi < -500:
            f -= (-500.0 + math.fmod(abs(zi), 500.0)) * math.sin(
                math.sqrt(500.0 - math.fmod(abs(zi), 500.0))
            )
            tmp = (zi + 500.0) / 100.0
            f += tmp * tmp / nx
        else:
            f -= zi * math.sin(math.sqrt(abs(zi)))
    f = 4.189828872724338e+002 * nx + f
    return f


def katsuura_func(x, nx, Os):
    y = shiftfunc(x, Os)
    y = [v * 5.0 / 100.0 for v in y]
    z2 = [y[i] * math.pow(100.0, 1.0 * i / (nx - 1) / 2.0) for i in range(nx)]
    tmp3 = math.pow(1.0 * nx, 1.2)
    f = 1.0
    for i in range(nx):
        temp = 0.0
        for j in range(1, 33):
            tmp1 = math.pow(2.0, j)
            tmp2 = tmp1 * z2[i]
            temp += abs(tmp2 - math.floor(tmp2 + 0.5)) / tmp1
        f *= math.pow(1.0 + (i + 1) * temp, 10.0 / tmp3)
    tmp1 = 10.0 / (nx * nx)
    f = f * tmp1 - tmp1
    return f


def bi_rastrigin_func(x, nx, Os):
    mu0 = 2.5
    d = 1.0
    s = 1.0 - 1.0 / (2.0 * math.sqrt(nx + 20.0) - 8.2)
    mu1 = -math.sqrt((mu0 * mu0 - d) / s)

    y = shiftfunc(x, Os)
    y = [v * 10.0 / 100.0 for v in y]
    tmpx = [0.0] * nx
    for i in range(nx):
        tmpx[i] = 2.0 * y[i]
        if Os[i] < 0.0:
            tmpx[i] *= -1.0
    z = tmpx[:]  # copy
    tmpx2 = [tmpx[i] + mu0 for i in range(nx)]
    for i in range(nx):
        z[i] *= math.pow(100.0, 1.0 * i / (nx - 1) / 2.0)

    tmp1 = 0.0
    tmp2 = 0.0
    for i in range(nx):
        t = tmpx2[i] - mu0
        tmp1 += t * t
        t = tmpx2[i] - mu1
        tmp2 += t * t
    tmp2 = tmp2 * s + d * nx

    tmp = 0.0
    for i in range(nx):
        tmp += math.cos(2.0 * PI * z[i])

    if tmp1 < tmp2:
        f = tmp1
    else:
        f = tmp2
    f += 10.0 * (nx - tmp)
    return f


def grie_rosen_func(x, nx, Os):
    y = shiftfunc(x, Os)
    y = [v * 5.0 / 100.0 for v in y]
    z = [v + 1.0 for v in y]
    f = 0.0
    for i in range(nx - 1):
        tmp1 = z[i] * z[i] - z[i + 1]
        tmp2 = z[i] - 1.0
        temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
        f += (temp * temp) / 4000.0 - math.cos(temp) + 1.0
    tmp1 = z[nx - 1] * z[nx - 1] - z[0]
    tmp2 = z[nx - 1] - 1.0
    temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
    f += (temp * temp) / 4000.0 - math.cos(temp) + 1.0
    return f


def escaffer6_func(x, nx, Os):
    y = shiftfunc(x, Os)
    y2 = asyfunc(y, 0.5)
    f = 0.0
    for i in range(nx - 1):
        temp1 = math.sin(math.sqrt(y2[i] * y2[i] + y2[i + 1] * y2[i + 1]))
        temp1 = temp1 * temp1
        temp2 = 1.0 + 0.001 * (y2[i] * y2[i] + y2[i + 1] * y2[i + 1])
        f += 0.5 + (temp1 - 0.5) / (temp2 * temp2)
    temp1 = math.sin(math.sqrt(y2[nx - 1] * y2[nx - 1] + y2[0] * y2[0]))
    temp1 = temp1 * temp1
    temp2 = 1.0 + 0.001 * (y2[nx - 1] * y2[nx - 1] + y2[0] * y2[0])
    f += 0.5 + (temp1 - 0.5) / (temp2 * temp2)
    return f


# --- wrapper for all functions ---


def test_func(x_flat, nx, mx, func_num, OShift):
    """
    Evaluate test functions without rotation.

    x_flat: list of length mx*nx, concatenated particles.
    Returns list of length mx with fitness values.
    """
    f = [0.0] * mx
    for i in range(mx):
        xi = x_flat[i * nx:(i + 1) * nx]
        if func_num == 1:
            f[i] = sphere_func(xi, nx, OShift)
        elif func_num == 2:
            f[i] = ellips_func(xi, nx, OShift)
        elif func_num == 3:
            f[i] = bent_cigar_func(xi, nx, OShift)
        elif func_num == 4:
            f[i] = discus_func(xi, nx, OShift)
        elif func_num == 5:
            f[i] = dif_powers_func(xi, nx, OShift)
        elif func_num == 6:
            f[i] = rosenbrock_func(xi, nx, OShift)
        elif func_num == 7:
            f[i] = schaffer_F7_func(xi, nx, OShift)
        elif func_num == 8:
            f[i] = ackley_func(xi, nx, OShift)
        elif func_num == 9:
            f[i] = weierstrass_func(xi, nx, OShift)
        elif func_num == 10:
            f[i] = griewank_func(xi, nx, OShift)
        elif func_num == 11:
            f[i] = rastrigin_func(xi, nx, OShift)
        elif func_num == 12:
            f[i] = rastrigin_func(xi, nx, OShift)  # Same as 11 without rotation
        elif func_num == 13:
            f[i] = step_rastrigin_func(xi, nx, OShift)
        elif func_num == 14:
            f[i] = schwefel_func(xi, nx, OShift)
        elif func_num == 15:
            f[i] = schwefel_func(xi, nx, OShift)  # Same as 14 without rotation
        elif func_num == 16:
            f[i] = katsuura_func(xi, nx, OShift)
        elif func_num == 17:
            f[i] = bi_rastrigin_func(xi, nx, OShift)
        elif func_num == 18:
            f[i] = bi_rastrigin_func(xi, nx, OShift)  # Same as 17 without rotation
        elif func_num == 19:
            f[i] = grie_rosen_func(xi, nx, OShift)
        elif func_num == 20:
            f[i] = escaffer6_func(xi, nx, OShift)
        else:
            raise ValueError(f"Unknown func_num: {func_num}")
    return f


def main():
    T_max = 500      # PSO run time
    rr = 10          # trials
    nx = 5           # dimension
    np = 20          # number of particles
    cf_num = 10
    sz = 200.0

    # allocate arrays (same shapes as C code)
    x = [0.0] * (np * nx)
    v = [0.0] * (np * nx)
    px = [0.0] * (np * nx)
    gx = [0.0] * nx
    pv = [0.0] * (np * nx)
    gv = [0.0] * nx

    fx = [0.0] * np
    fv = [0.0] * np
    pfx = [0.0] * np
    pfv = [0.0] * np

    # --- read shift data shift_data.txt ---
    s_file = "./inst/extdata/shift_data.txt"
    try:
        with open(s_file, "r") as fpt:
            tokens = fpt.read().split()
    except OSError:
        raise RuntimeError(f"Cannot open input file for reading: {s_file}")

    expected_O = cf_num * nx
    if len(tokens) < expected_O:
        raise RuntimeError(
            f"Not enough values in {s_file}: expected {expected_O}, got {len(tokens)}"
        )
    OShift = [float(tok) for tok in tokens[:expected_O]]

    # --- output files ---
    grid_file = "PSO_grid_per_func.txt"
    summary_file = "PSO_best_params.txt"

    with open(grid_file, "w") as grid_out, open(summary_file, "w") as summary_out:
        # loop over functions 1..20
        for func_num in range(1, 21):
            print(f"Optimising PSO params for function {func_num}")
            best_mean = INF
            best_a = None
            best_w = None

            w = -1.15
            while w < 1.150001:
                a = 0.05
                while a < 6.05:
                    ga = 0.0
                    a1 = 0.5 * a
                    a2 = a - a1

                    # rr independent PSO runs for this (func_num, a, w)
                    for _r in range(rr):
                        gfx = 1.0e15
                        gfv = 1.0e15

                        # initialise swarm
                        for i in range(np * nx):
                            v[i] = (random.random() - 0.5) * 2.0
                            x[i] = (random.random() - 0.5) * sz
                            px[i] = x[i]
                        for j in range(nx):
                            gx[j] = (random.random() - 0.5) * sz
                            gv[j] = (random.random() - 0.5) * 2.0

                        # reset personal best fitness for this run
                        for i in range(np):
                            pfx[i] = INF
                            pfv[i] = INF

                        oo = False
                        t = 0
                        while t < T_max:
                            fx = test_func(x, nx, np, func_num, OShift)
                            fv = test_func(v, nx, np, func_num, OShift)

                            # update personal / global best for x
                            for n in range(np):
                                if fx[n] < pfx[n]:
                                    pfx[n] = fx[n]
                                    base = n * nx
                                    for j in range(nx):
                                        px[base + j] = x[base + j]
                                if fx[n] < gfx:
                                    gfx = fx[n]
                                    base = n * nx
                                    for j in range(nx):
                                        gx[j] = x[base + j]

                            # update personal / global best for v
                            for n in range(np):
                                if fv[n] < pfv[n]:
                                    pfv[n] = fv[n]
                                    base = n * nx
                                    for j in range(nx):
                                        pv[base + j] = v[base + j]
                                if fv[n] < gfv:
                                    gfv = fv[n]
                                    base = n * nx
                                    for j in range(nx):
                                        gv[j] = v[base + j]

                            # velocity and position update
                            for i in range(np * nx):
                                v[i] = (
                                    w * v[i]
                                    + a1 * random.random() * (px[i] - x[i])
                                    + a2 * random.random() * (gx[i % nx] - x[i])
                                )
                                x[i] += v[i]

                                if abs(x[i]) > 1.0e6 or abs(v[i]) > 1.0e6:
                                    oo = True

                            if oo:
                                break
                            t += 1

                        # accumulate best-of-position vs best-of-velocity for this run
                        ga += gfx if gfx < gfv else gfv

                    # average over rr runs for this (func_num, a, w)
                    mean_ga = ga / rr

                    # write full grid: func_num, a, w, mean_best
                    grid_out.write(f"{func_num} {a} {w} {mean_ga}\n")

                    # track best (a, w) for this function
                    if mean_ga < best_mean:
                        best_mean = mean_ga
                        best_a = a
                        best_w = w

                    a += 0.1

                grid_out.write("\n")  # separate w slices for readability
                grid_out.flush()
                w += 0.05

            # after all (a, w) for this function, write best config
            summary_out.write(
                f"func {func_num}: best_a={best_a} best_w={best_w} mean_best={best_mean}\n"
            )
            summary_out.flush()


if __name__ == "__main__":
    main()