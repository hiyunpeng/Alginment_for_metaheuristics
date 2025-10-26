# Dual-ascent with heuristic repair for 0-1 Knapsack

from typing import List, Tuple, Optional
import math
import random

Item = Tuple[int, int]  # (value, weight)


def argmin_lagrangian_binary(items: List[Item], lam: float) -> List[int]:
    """
    Exact minimization of L(x, lam) over x in {0,1}^n:
      L = -lam*C + sum_i (v_i - lam*w_i) x_i.
    For fixed lam, choose x_i=1 iff (v_i - lam*w_i) > 0 (ties -> 0).
    Returns a binary list x.
    """
    x = []
    for v, w in items:
        r = v - lam * w
        x.append(1 if r > 0 else 0)
    return x


def total_weight(items: List[Item], x: List[int]) -> int:
    return sum(w for (bit, (_, w)) in zip(x, items) if bit)


def total_value(items: List[Item], x: List[int]) -> int:
    return sum(v for (bit, (v, _)) in zip(x, items) if bit)


def dual_value_phi(items: List[Item], lam: float, capacity: int) -> float:
    # phi(lam) = -lam*C + sum_i max(0, v_i - lam*w_i)
    s = -lam * capacity
    for v, w in items:
        s += max(0.0, v - lam * w)
    return s


def local_swap_improve(items: List[Item], x: List[int], capacity: int, max_trials: int = 200) -> List[int]:
    """
    Simple randomized 1-1 swap hill climb: try swapping an included item with an excluded item
    if it fits and improves value. Limited trials for speed.
    """
    n = len(items)
    best_x = x[:]
    best_v = total_value(items, best_x)

    incl = [i for i in range(n) if best_x[i] == 1]
    excl = [i for i in range(n) if best_x[i] == 0]

    # If small instance, we can try all pairs; else random trials
    if len(incl) * len(excl) <= max_trials:
        pairs = [(i, j) for i in incl for j in excl]
    else:
        pairs = []
        for _ in range(max_trials):
            if not incl or not excl:
                break
            i = random.choice(incl)
            j = random.choice(excl)
            pairs.append((i, j))

    base_w = total_weight(items, best_x)
    for i, j in pairs:
        vi, wi = items[i]
        vj, wj = items[j]
        new_w = base_w - wi + wj
        if new_w <= capacity:
            new_v = best_v - vi + vj
            if new_v > best_v:
                # Accept improvement
                best_x[i] = 0
                best_x[j] = 1
                best_v = new_v
                base_w = new_w
                # Update incl/excl pools
                if i in incl:
                    incl.remove(i)
                if j in excl:
                    excl.remove(j)

    return best_x


def repair_to_capacity(items: List[Item], x: List[int], lam: float, capacity: int) -> List[int]:
    """
    Greedy drop guided by reduced-profit-per-weight, then optional 1-1 swap improvement.
    """
    n = len(items)
    w = total_weight(items, x)
    if w <= capacity:
        repaired = x[:]
    else:
        # Build list of included items with their reduced-profit-per-weight (lower is worse -> drop first)
        inc = [(i, (items[i][0] - lam * items[i][1]) / items[i][1]) for i in range(n) if x[i] == 1]
        # Sort ascending (worst first)
        inc.sort(key=lambda t: t[1])
        repaired = x[:]
        for i, _ in inc:
            if total_weight(items, repaired) <= capacity:
                break
            repaired[i] = 0
        # Ensure feasible
        while total_weight(items, repaired) > capacity:
            # If still overweight due to ties/rounding, drop the heaviest remaining included item
            incl_idxs = [i for i in range(n) if repaired[i] == 1]
            if not incl_idxs:
                break
            i_heavy = max(incl_idxs, key=lambda i: items[i][1])
            repaired[i_heavy] = 0

    # Local 1-1 swap improvement: try adding one excluded and dropping one included to improve value
    repaired = local_swap_improve(items, repaired, capacity)
    return repaired


def subgradient_step(t: int, g: float, base_step: float = 1.0, schedule: str = "diminishing") -> float:
    if schedule == "diminishing":
        return base_step / math.sqrt(t + 1) * g
    elif schedule == "polyak":
        # Placeholder Polyak step: requires best known UB and dual value to estimate step; use diminishing as fallback
        return base_step / math.sqrt(t + 1) * g
    else:
        return 0.1 * g


def solve_knapsack_dual_meta(
    items: List[Item],
    capacity: int,
    T: int = 200,
    lam0: float = 0.0,
    step_base: float = 1.0,
    gap_tol: float = 1e-3,
    seed: int = 0
):
    random.seed(seed)
    lam = max(0.0, lam0)
    best_feasible: Optional[List[int]] = None
    UB = float("-inf")
    LB = float("-inf")
    history = []

    for t in range(T):
        # 1) Exact inner solve for the Lagrangian relaxation at current lambda
        x_relax = argmin_lagrangian_binary(items, lam)

        # 2) Heuristic repair to feasibility + local swap
        x_feas = repair_to_capacity(items, x_relax, lam, capacity)
        v_feas = total_value(items, x_feas)
        if v_feas > UB:
            UB = v_feas
            best_feasible = x_feas[:]

        # 3) Dual bound (valid since inner solve is exact for this relaxation)
        LB = max(LB, dual_value_phi(items, lam, capacity))

        # 4) Subgradient ascent on lambda
        viol = total_weight(items, x_relax) - capacity  # overweight
        step = subgradient_step(t, viol, base_step=step_base, schedule="diminishing")
        lam = max(0.0, lam + step)

        # 5) Telemetry
        gap = (UB - LB) / (abs(UB) + 1e-9) if UB > -1e18 and LB < 1e18 else float("inf")
        history.append((t, lam, UB, LB, viol, gap))

        # Stopping rule
        if abs(viol) < 1e-9 and gap <= gap_tol:
            break

    return {
        "best_x": best_feasible,
        "UB": UB,
        "LB": LB,
        "lambda": lam,
        "history": history,
    }


def demo():
    # Example instance from the discussion
    ITEMS = [
        (10, 9),  # (value, weight)
        (8, 6),
        (7, 5),
        (6, 4),
    ]
    CAPACITY = 10

    result = solve_knapsack_dual_meta(
        ITEMS, CAPACITY, T=200, lam0=0.0, step_base=0.5, gap_tol=1e-6, seed=42
    )

    print("=== Dual + Metaheuristic Repair (0-1 Knapsack) ===")
    print(f"Items (v,w): {ITEMS}")
    print(f"Capacity   : {CAPACITY}")
    print(f"Best x     : {result['best_x']}")
    print(f"UB (value) : {result['UB']}")
    print(f"LB (dual)  : {result['LB']:.4f}")
    print(f"lambda_end : {result['lambda']:.4f}")
    print(f"Gap        : {(result['UB'] - result['LB'])/(abs(result['UB'])+1e-9):.4f}")

    # Show a short tail of the history
    print("\n[t]   lambda     UB      LB      viol   gap")
    for t, lam, UB, LB, viol, gap in result["history"][-10:]:
        print(f"{t:3d}  {lam:7.4f}  {UB:6.2f}  {LB:7.2f}  {viol:6.2f}  {gap:6.3f}")


# Run demo
demo()
