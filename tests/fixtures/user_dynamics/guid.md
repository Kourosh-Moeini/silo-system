# User Dynamics Regression Fixtures

This directory allows teammates to easily add custom `.py` plant models to our automated testing suite **without writing any new test code**.

## How to Add a New Model to CI

1. **Drop your file in this folder:** 
   Add your Python file containing the `dynamics(t, x, u)` function directly into `tests/fixtures/user_dynamics/`.
   *(e.g., `magnetic_robot.py`)*

2. **Update `registry.json`:**
   Open `registry.json` and append a new JSON object for your file. You must provide basic "smoke" gains that will allow the simulation to run to completion without math errors (like diverging to infinity instantly).

   ```json
   {
     "name": "Magnetic Continuum Robot",
     "path": "magnetic_robot.py",
     "num_states": 4,
     "smoke_gains": {
       "Kp": 5.0,
       "Kd": 0.1
     },
     "default_scenario": {
       "initial_condition_range": [-0.5, 0.5],
       "randomness_level": 0.0,
       "disturbance_level": 0.0
     }
   }