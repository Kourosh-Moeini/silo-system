"""
probe_complexity.py

PRACTICE / EXPERIMENT SCRIPT — an intrinsic *plant* complexity rate (0..1) per
built-in plant. This is the sandbox where the plant-complexity methodology for
the scenario report is being developed. It is NOT collected by pytest (the file
name does not start with `test_`); run it directly:

    python tests/probe_complexity.py

It scores each plant with three independent open-loop probes and blends them:

  1) TARGET-HOLD  ("does it stay where you put it?")
     Place the initial condition exactly on a non-zero output target, turn the
     controller OFF (u = 0), and let the plant run free. A plant that holds an
     arbitrary setpoint on its own (e.g. a pure integrator) barely moves; a
     plant that is pulled off the setpoint (self-regulating or unstable) moves.
         score = mean_t |output(t) - target| / |target|, averaged over targets.
         ~0 -> stays put (easy) ;  larger -> won't hold the setpoint (harder).

  2) NONLINEARITY  ("does it obey superposition?")
     Fix a single initial position on the output channel and apply a constant
     control signal u. A LINEAR plant is homogeneous: scaling (x0, u) by a
     factor a scales the whole trajectory by exactly a. A NONLINEAR plant does
     not. We measure how far the scaled runs drift from a * (base run):
         score = mean_a || sim(a*x0, a*u) - a*sim(x0, u) || / || a*sim(x0, u) ||.
         ~0 -> linear (easy) ;  larger -> nonlinear (harder).

  3) SWING  ("does it oscillate when nudged at the target?")
     Place the initial condition on the target and apply a SMALL constant
     control action, then watch the free-ish response. An oscillatory
     (underdamped) plant swings back and forth; a well-damped / monotonic plant
     just slides to a new value. We count direction reversals (turning points)
     of the output = the number of swings:
         score = mean_target (# velocity sign changes over the horizon).
         0 -> monotonic, no oscillation (easy) ;  more -> oscillatory (harder).

Each raw score is min-max normalized to 0..1 across the plants (easiest -> 0,
hardest -> 1) and blended with configurable weights into one complexity rate.

WHY THE ENGINE DIRECTLY (not POST /silo/simulate):
The simulate endpoint only runs CLOSED-LOOP (u is computed from the gains and
the tracking error); it cannot inject an arbitrary open-loop control. The
linearity and swing probes need a *known applied* u, so we drive the plant
dynamics directly through `create_system(...).system_dynamics(x, u)` with the
same forward-Euler step the engine uses. Driving it ourselves also lets us
bypass control saturation while probing.

#states / #actions are printed only for reference: they feed the SCENARIO-INPUT
complexity (distance-to-target, randomness, disturbance, num_states,
num_actions), which is handled separately in the test/report layer.
"""

import sys
from pathlib import Path

import numpy as np

# --- make the repo root importable no matter where this file sits ----------
_root = Path(__file__).resolve().parent
while _root != _root.parent and not (_root / "api").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root))

from src.systems import create_system


# --- probe configuration ---------------------------------------------------
PLANTS = ["ball_beam", "dc_motor", "inverted_pendulum"]

PROBE_DT = 0.001          # small step so fast modes (e.g. the DC motor) stay stable
PROBE_MAX_TIME = 2.0      # long enough to reveal drift / a full oscillation

# Output setpoints used by BOTH the target-hold probe (u = 0) and the swing
# probe (small u): in both we start the plant exactly on the target.
TARGETS = [0.25, 0.5, 1.0]

# SWING probe: the small control action applied while sitting on the target.
SWING_U = 0.1

# NONLINEARITY probe: one initial output position, one base control, several scales.
LIN_INIT_OUTPUT = 0.2
LIN_BASE_U = 1.0
LIN_SCALES = [0.25, 0.5, 2.0, 4.0]

# Blend weights for the three probes (must sum to 1).
WEIGHTS = {"target_hold": 1 / 3, "nonlinearity": 1 / 3, "swing": 1 / 3}


# --- engine-direct open-loop simulation ------------------------------------
def _make_system(plant: str):
    """Build a built-in plant and pin the probe's dt / horizon."""
    system = create_system(plant)
    system.dt = PROBE_DT
    system.max_time = PROBE_MAX_TIME
    return system


def _simulate(system, x0, u_const: float, n_steps: int) -> np.ndarray:
    """Forward-Euler open-loop rollout applying a CONSTANT control `u_const`.

    Returns the full state trajectory, shape (n_steps + 1, num_states). No
    control clipping and no divergence break: we want the raw plant response."""
    x = np.asarray(x0, dtype=float).copy()
    traj = np.empty((n_steps + 1, system.num_states), dtype=float)
    traj[0] = x
    for i in range(n_steps):
        x_dot = np.asarray(system.system_dynamics(x, float(u_const)), dtype=float)
        x = x + x_dot * system.dt
        traj[i + 1] = x
    return traj


# --- probe 1: target-hold ("does it stay where you put it?") ---------------
def measure_target_hold(system, n_steps: int) -> float:
    """Release the plant at each target with u = 0; average the normalized
    distance it drifts away from that target over the horizon."""
    oc = system.output_channel
    scores = []
    for tgt in TARGETS:
        x0 = np.zeros(system.num_states)
        x0[oc] = tgt
        out = _simulate(system, x0, 0.0, n_steps)[:, oc]
        scores.append(float(np.mean(np.abs(out - tgt)) / (abs(tgt) + 1e-12)))
    return float(np.mean(scores)) if scores else 0.0


# --- probe 2: nonlinearity (homogeneity / superposition) -------------------
def measure_nonlinearity(system, n_steps: int) -> float:
    """From one initial position, apply a constant u and test homogeneity:
    for a linear plant sim(a*x0, a*u) == a*sim(x0, u). Average the relative
    deviation over the scale factors."""
    oc = system.output_channel
    x0 = np.zeros(system.num_states)
    x0[oc] = LIN_INIT_OUTPUT
    base = _simulate(system, x0, LIN_BASE_U, n_steps)

    devs = []
    for a in LIN_SCALES:
        scaled = _simulate(system, a * x0, a * LIN_BASE_U, n_steps)
        ref = a * base
        denom = float(np.linalg.norm(ref)) + 1e-12
        devs.append(float(np.linalg.norm(scaled - ref) / denom))
    return float(np.mean(devs)) if devs else 0.0


# --- probe 3: swing ("does it oscillate when nudged at the target?") -------
def _count_reversals(signal: np.ndarray) -> int:
    """Number of direction reversals (turning points) of a 1-D signal, i.e.
    how many times its slope changes sign. Flat steps are ignored so numerical
    jitter does not manufacture swings. Monotonic -> 0; oscillation -> many."""
    d = np.diff(signal)
    s = np.sign(d)
    s = s[s != 0]                       # drop flat / zero-velocity steps
    if s.size < 2:
        return 0
    return int(np.sum(s[1:] != s[:-1]))


def measure_swing(system, n_steps: int) -> float:
    """Sit on each target, apply a SMALL constant control action, and count the
    output's direction reversals (swings), averaged over the targets."""
    oc = system.output_channel
    counts = []
    for tgt in TARGETS:
        x0 = np.zeros(system.num_states)
        x0[oc] = tgt
        out = _simulate(system, x0, SWING_U, n_steps)[:, oc]
        counts.append(_count_reversals(out))
    return float(np.mean(counts)) if counts else 0.0


# --- normalization + blend -------------------------------------------------
def _normalize(values: dict) -> dict:
    """Min-max to 0..1 across plants: easiest -> 0, hardest -> 1. All equal -> 0."""
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def compute_plant_complexity(plants=None) -> dict:
    """Run all three probes on `plants` and return the full breakdown.

    This is the PUBLIC entry point reused by tests/make_report.py so the
    complexity methodology lives in exactly one place.

    Returns
    -------
    {
      "plants":       [...],
      "raw":          {probe: {plant: raw_score}},
      "normalized":   {probe: {plant: 0..1}},
      "complexity":   {plant: 0..1},          # blended, min-max across plants
      "num_states":   {plant: int},
      "num_actions":  {plant: int},
      "config":       {...}                   # probe settings, for the report
    }
    """
    plants = list(plants) if plants else list(PLANTS)
    n_steps = int(PROBE_MAX_TIME / PROBE_DT)
    systems = {p: _make_system(p) for p in plants}

    raw = {
        "target_hold":  {p: measure_target_hold(systems[p], n_steps) for p in plants},
        "nonlinearity": {p: measure_nonlinearity(systems[p], n_steps) for p in plants},
        "swing":        {p: measure_swing(systems[p], n_steps) for p in plants},
    }
    norm = {k: _normalize(raw[k]) for k in raw}
    complexity = {p: sum(WEIGHTS[k] * norm[k][p] for k in raw) for p in plants}

    return {
        "plants": plants,
        "raw": raw,
        "normalized": norm,
        "complexity": complexity,
        "num_states": {p: systems[p].num_states for p in plants},
        "num_actions": {p: getattr(systems[p], "num_inputs", 1) for p in plants},
        "config": {
            "dt": PROBE_DT,
            "max_time": PROBE_MAX_TIME,
            "targets": list(TARGETS),
            "swing_u": SWING_U,
            "lin_init_output": LIN_INIT_OUTPUT,
            "lin_base_u": LIN_BASE_U,
            "lin_scales": list(LIN_SCALES),
            "weights": dict(WEIGHTS),
        },
    }


def main() -> None:
    result = compute_plant_complexity(PLANTS)
    raw, norm = result["raw"], result["normalized"]
    complexity = result["complexity"]
    states, actions = result["num_states"], result["num_actions"]

    print(f"\nPlant complexity probe  (dt={PROBE_DT}, horizon={PROBE_MAX_TIME}s, "
          f"open-loop via engine dynamics)")
    print(f"  target-hold : release at targets {TARGETS} with the controller OFF")
    print(f"  nonlinearity: x0[out]={LIN_INIT_OUTPUT}, u={LIN_BASE_U}, scales={LIN_SCALES}")
    print(f"  swing       : sit on targets {TARGETS} with small u={SWING_U}, count reversals\n")

    print("Raw measurements")
    print(f"{'plant':<20}{'target_hold':>13}{'nonlinearity':>14}{'swing':>9}"
          f"{'#states':>9}{'#actions':>10}")
    print("-" * 75)
    for p in PLANTS:
        print(f"{p:<20}{raw['target_hold'][p]:>13.4f}{raw['nonlinearity'][p]:>14.4f}"
              f"{raw['swing'][p]:>9.1f}{states[p]:>9}{actions[p]:>10}")

    print("\nNormalized (0..1) and blended complexity rate")
    print(f"{'plant':<20}{'hold':>9}{'nonlin.':>9}{'swing':>9}{'complexity':>14}")
    print("-" * 61)
    for p in sorted(PLANTS, key=lambda k: complexity[k]):
        print(f"{p:<20}{norm['target_hold'][p]:>9.2f}{norm['nonlinearity'][p]:>9.2f}"
              f"{norm['swing'][p]:>9.2f}{complexity[p]:>14.3f}")

    print(f"\n(0 = easiest, 1 = hardest; weights "
          f"target_hold={WEIGHTS['target_hold']:.2f}, "
          f"nonlinearity={WEIGHTS['nonlinearity']:.2f}, "
          f"swing={WEIGHTS['swing']:.2f})")
    print("(#states / #actions are shown for the scenario-input complexity, "
          "handled in the report layer.)")


if __name__ == "__main__":
    main()
