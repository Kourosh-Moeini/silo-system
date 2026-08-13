import json
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import importlib.util
from typing import Dict, List, Any, Tuple, Optional, TypedDict
from datetime import datetime
from pathlib import Path
import os


class GeneralDynamicalSystem:
    """Base class for dynamical systems with arbitrary number of states"""

    def __init__(self, scenario=None):
        # System identification
        self.name = "General Dynamical System"
        self.description = "Override in subclass"

        # Simulation parameters (configurable)
        self.dt = 0.01
        self.max_time = 10.0
        self.min_control = -10.0
        self.max_control = 10.0
        self.target = 0.0
        self.num_inputs = 1
        self.input_channel = 0
        self.output_channel = 0
        self.trim_values = np.zeros(1)  # Will be resized appropriately

        # System-specific parameters (to be defined in subclasses)
        self.num_states = 0
        self.state_names = []
        self.control_input_names = []
        self.num_controls = 0

        # Performance thresholds (system-specific)
        self.failure_conditions = {}
        self.max_control_limits = {}

    def get_control_param_schema(self, controller_type):
        """Return the parameter schema for a given controller type"""
        if controller_type == "FSF":
            # Full-state feedback: one gain per state
            return {f"K{i + 1}": {"min": 0.1, "max": 10.0}
                    for i in range(self.num_states)}
        elif controller_type in ["P", "PI", "PD", "PID"]:
            # PID controllers work with output feedback (typically first state)
            schema = {}
            if controller_type in ["P", "PI", "PD", "PID"]:
                schema["Kp"] = {"min": 0.1, "max": 20.0}
            if controller_type in ["PI", "PID"]:
                schema["Ki"] = {"min": 0.0, "max": 5.0}
            if controller_type in ["PD", "PID"]:
                schema["Kd"] = {"min": 0.0, "max": 5.0}
            return schema
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

    def run_simulation(self, control_params):
        """Generalized simulation runner"""
        # Determine controller type based on parameters
        if isinstance(control_params, dict):
            # Check if this is a full-state feedback controller
            fsf_keys = [f"K{i + 1}" for i in range(self.num_states)]
            if any(key in control_params for key in fsf_keys):
                # Full-state feedback controller
                K_values = [control_params.get(f"K{i + 1}", 0.0)
                            for i in range(self.num_states)]
                return self.run_fsf_simulation(K_values)
            else:
                # PID-type controller
                Kp = control_params.get('Kp', 0.0)
                Ki = control_params.get('Ki', 0.0)
                Kd = control_params.get('Kd', 0.0)
                return self.run_pid_simulation(Kp, Ki, Kd)
        else:
            # Handle legacy tuple/list format
            if len(control_params) == self.num_states:
                # Assume FSF gains
                return self.run_fsf_simulation(control_params)
            else:
                # Assume PID parameters
                Kp, Ki, Kd = control_params[:3]
                return self.run_pid_simulation(Kp, Ki, Kd)

    def run_fsf_simulation(self, K_values):
        """Full-state feedback simulation - to be implemented in subclass"""
        raise NotImplementedError("Implement in subclass")

    def run_pid_simulation(self, Kp, Ki, Kd):
        """PID simulation - to be implemented in subclass"""
        raise NotImplementedError("Implement in subclass")


class CustomDynamicalSystem(GeneralDynamicalSystem):
    """Generic SISO system from user-uploaded dynamics"""

    def __init__(self, dynamics_file_path, scenario=None, num_inputs: int = 1, name: Optional[str] = None,
                 description: Optional[str] = None):
        super().__init__(scenario)
        self.num_inputs = num_inputs

        # Load the user dynamics
        self.dynamics_file_path = dynamics_file_path
        self._load_dynamics_module()

        # Detect system properties
        self._detect_system_properties()

        # Set system identification
        if name:
            self.name = name
        else:
            self.name = "Custom SISO System"
        if description:
            self.description = description
        else:
            self.description = f"User-defined {self.num_states}-state SISO system from uploaded dynamics"

        # Default simulation parameters (can be overridden)
        self.dt = 0.01
        self.max_time = 10.0
        self.min_control = -10.0
        self.max_control = 10.0
        self.target = 0.0
        self.target = 0.0
        self.num_inputs = 1
        self.input_channel = 0
        self.output_channel = 0
        self.trim_values = np.zeros(1)

        # Generic state names
        self.state_names = [f'x{i}' for i in range(self.num_states)]
        self.control_input_names = ['u']
        self.num_controls = 1

        # Apply scenario parameters if provided
        self.initial_condition_range = [-1.0, 1.0]
        self.randomness_level = 0.0
        self.disturbance_level = 0.0
        self.param_uncertainty = 0.0

        if scenario:
            self.apply_scenario(scenario)

    def _load_dynamics_module(self):
        """Load the dynamics function from the uploaded file"""
        spec = importlib.util.spec_from_file_location("user_dynamics", self.dynamics_file_path)
        self.dynamics_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.dynamics_module)
        self.dynamics_func = getattr(self.dynamics_module, 'dynamics')

    def _detect_system_properties(self):
        """Auto-detect the number of states by testing the dynamics function"""
        test_t = 0.0
        test_u = 0.0 if self.num_inputs == 1 else np.zeros(self.num_inputs)

        # Try different state dimensions to find the correct one
        for n_states in range(1, 11):
            try:
                test_x = np.zeros(n_states)
                result = self.dynamics_func(test_t, test_x, test_u)
                result = np.asarray(result)

                if result.shape == (n_states,):
                    self.num_states = n_states
                    return
            except:
                continue

        raise ValueError("Could not determine system dimension from dynamics function")

    def apply_scenario(self, scenario_params):
        """Apply scenario parameters"""
        ic_range = scenario_params.get('initial_condition_range', [-1.0, 1.0])
        self.initial_condition_range = ic_range

        self.randomness_level = scenario_params.get('randomness_level', 0.0)
        self.disturbance_level = scenario_params.get('disturbance_level', 0.0)
        self.param_uncertainty = scenario_params.get('param_uncertainty', 0.0)

    def system_dynamics(self, x, u):
        """Wrapper for user-provided dynamics function

        Args:
            x: State vector
            u: Control input (scalar for SISO or vector for MIMO)
        """
        t = 0.0  # Time-invariant assumption for simplicity

        # Ensure u is properly formatted for the dynamics function
        if self.num_inputs == 1:
            u_input = u[0] if isinstance(u, (list, np.ndarray)) else float(u)
        else:
            u_input = u

        return self.dynamics_func(t, x, u_input)

    def run_pid_simulation(self, Kp, Ki, Kd, initial_state=None):
        """PID control simulation for custom system

        Args:
            Kp, Ki, Kd: PID gains
            initial_state: Optional fixed initial state. If None, random IC is used.
        """
        expected_steps = int(self.max_time / self.dt) + 1
        t = np.arange(0, self.max_time + self.dt, self.dt)[:expected_steps]
        n = len(t)

        # Initialize state
        if initial_state is not None:
            x = initial_state.copy()  # Use provided initial state
        else:
            # Random initial conditions (original behavior)
            x = np.zeros(self.num_states)
            x[self.output_channel] = np.random.uniform(*self.initial_condition_range)

        # Rest of the method remains the same...
        # PID variables
        integral = 0.0
        prev_error = 0.0

        # History tracking
        output_history = [x[self.output_channel]]
        u_history = []
        errors = []

        for i in range(n):
            # Get current output and calculate error
            output = x[self.output_channel]

            # Handle target (can be scalar or callable for time-varying targets)
            current_target = self.target(i * self.dt) if callable(self.target) else self.target
            error = current_target - output

            # Add measurement noise
            if self.randomness_level > 0:
                error += np.random.normal(0, self.randomness_level)

            # PID control
            integral += error * self.dt
            derivative = (error - prev_error) / self.dt if i > 0 else 0.0
            u_control = Kp * error + Ki * integral + Kd * derivative

            # Add disturbance
            if self.disturbance_level > 0:
                u_control += self.disturbance_level * np.sin(2 * np.pi * 5 * i * self.dt)

            # Apply control limits
            u_control = np.clip(u_control, self.min_control, self.max_control)

            # Build full control input vector
            u_full = self.trim_values.copy()
            u_full[self.input_channel] = u_control

            # Record history (only the controlled input)
            u_history.append(u_control)
            errors.append(error)

            # State propagation using Euler integration
            try:
                x_dot = self.system_dynamics(x, u_full)
                x = x + x_dot * self.dt
                output_history.append(x[self.output_channel])
                prev_error = error
            except:
                # If simulation becomes unstable, break
                break

        # Trim to actual simulation length
        output_history = np.array(output_history[:-1])
        u_history = np.array(u_history)
        errors = np.array(errors)

        return output_history, u_history, errors

    def run_fsf_simulation(self, K_values, initial_state=None):
        """Full-state feedback simulation for custom system

        Args:
            K_values: State feedback gains
            initial_state: Optional fixed initial state. If None, random IC is used.
        """
        expected_steps = int(self.max_time / self.dt) + 1
        t = np.arange(0, self.max_time + self.dt, self.dt)[:expected_steps]
        n = len(t)

        # Initialize state
        if initial_state is not None:
            x = initial_state.copy()  # Use provided initial state
        else:
            # Random initial conditions (original behavior)
            x = np.zeros(self.num_states)
            x[self.output_channel] = np.random.uniform(*self.initial_condition_range)

        # Rest of the method remains the same...
        # History tracking
        output_history = [x[self.output_channel]]
        u_history = []
        errors = []

        for i in range(n):
            # Apply measurement noise
            noise = np.random.normal(0, self.randomness_level,
                                     self.num_states) if self.randomness_level > 0 else np.zeros(self.num_states)
            x_noisy = x + noise

            # Get current target
            current_target = self.target(i * self.dt) if callable(self.target) else self.target

            # Error is the output channel deviation from target
            error = current_target - x_noisy[self.output_channel]
            errors.append(error)

            # Full-state feedback control law: u = -K^T * (x - x_desired)
            # We want to regulate the output channel to the target, other states to zero
            x_desired = np.zeros(self.num_states)
            x_desired[self.output_channel] = current_target
            state_error = x_noisy - x_desired

            u_control = -np.dot(K_values, state_error)

            # Add disturbance
            if self.disturbance_level > 0:
                u_control += self.disturbance_level * np.sin(2 * np.pi * 5 * i * self.dt)

            # Apply control limits
            u_control = np.clip(u_control, self.min_control, self.max_control)

            # Build full control input vector
            u_full = self.trim_values.copy()
            u_full[self.input_channel] = u_control

            u_history.append(u_control)

            # State propagation
            try:
                x_dot = self.system_dynamics(x, u_full)
                x = x + x_dot * self.dt
                output_history.append(x[self.output_channel])
            except:
                # If simulation becomes unstable, break
                break

        # Trim to actual simulation length
        output_history = np.array(output_history[:-1])
        u_history = np.array(u_history)
        errors = np.array(errors)

        return output_history, u_history, errors

    def plot_time_response(self, control_params, save_path=None, num_runs=50):
        """Plot time response for custom system"""
        # Determine controller type
        if isinstance(control_params, dict):
            if all(f"K{i + 1}" in control_params for i in range(self.num_states)):
                controller_type = "FSF"
                title_params = ", ".join(
                    [f"K{i + 1}={control_params[f'K{i + 1}']:.2f}" for i in range(self.num_states)])
            else:
                controller_type = "PID"
                title_params = f"Kp={control_params.get('Kp', 0):.2f}, Ki={control_params.get('Ki', 0):.2f}, Kd={control_params.get('Kd', 0):.2f}"
        else:
            controller_type = "Unknown"
            title_params = str(control_params)

        # Reconstruct scenario parameters
        scenario_params = {
            'initial_condition_range': self.initial_condition_range,
            'randomness_level': self.randomness_level,
            'disturbance_level': self.disturbance_level,
            'param_uncertainty': self.param_uncertainty
        }

        # Monte Carlo simulation
        all_outputs, all_controls = [], []
        for _ in range(num_runs):
            try:
                system = CustomDynamicalSystem(self.dynamics_file_path, scenario_params)
                output, control, _ = system.run_simulation(control_params)
                all_outputs.append(output)
                all_controls.append(control)
            except:
                continue  # Skip failed runs

        if not all_outputs:
            print("All simulation runs failed!")
            return

        # Process trajectories
        max_len = max(len(out) for out in all_outputs)
        time_vector = np.arange(0, max_len * self.dt, self.dt)

        output_matrix = np.full((len(all_outputs), max_len), np.nan)
        control_matrix = np.full((len(all_controls), max_len), np.nan)

        for i in range(len(all_outputs)):
            output_matrix[i, :len(all_outputs[i])] = all_outputs[i]
            control_matrix[i, :len(all_controls[i])] = all_controls[i]

        # Calculate statistics
        output_mean = np.nanmean(output_matrix, axis=0)
        output_std = np.nanstd(output_matrix, axis=0)
        control_mean = np.nanmean(control_matrix, axis=0)
        control_std = np.nanstd(control_matrix, axis=0)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Output plot
        ax1.plot(time_vector, output_mean, 'b-', lw=2, label='Mean Output')
        ax1.fill_between(time_vector, output_mean - output_std, output_mean + output_std,
                         color='blue', alpha=0.2, label='±1 SD')
        ax1.axhline(0, color='r', linestyle='--', alpha=0.7, label='Target')
        ax1.set_ylabel('System Output')
        ax1.set_title(f"Custom System Response ({num_runs} runs) | {controller_type} | {title_params}")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')

        # Control signal plot
        ax2.plot(time_vector, control_mean, 'g-', lw=2, label='Mean Control')
        ax2.fill_between(time_vector, control_mean - control_std, control_mean + control_std,
                         color='green', alpha=0.2, label='±1 SD')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Control Input')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Built-in name -> case_studies relative path (resolved from repo root)
_BUILTIN_CASE_STUDIES = {
    "ball_beam": "case_studies/BallBeam.py",
    "dc_motor": "case_studies/DCMotor.py",
    "inverted_pendulum": "case_studies/InvPendulum.py",
    "inv_pendulum": "case_studies/InvPendulum.py",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_case_study(system_name: str) -> Optional[Path]:
    key = (system_name or "").lower().replace("-", "_").replace(" ", "_")
    rel = _BUILTIN_CASE_STUDIES.get(key)
    if not rel:
        return None
    path = _repo_root() / rel
    return path if path.is_file() else None


def create_system(
    system_name: str,
    scenario=None,
    custom_dynamics_path: Optional[str] = None,
    num_inputs: int = 1,
    **_ignored,
):
    """Factory: custom path or built-in case_studies files via CustomDynamicalSystem.

    Built-in names (ball_beam, dc_motor, inverted_pendulum) load the matching
    file under case_studies/. Extra kwargs are ignored for API compatibility.
    """
    if custom_dynamics_path or system_name == "custom":
        if not custom_dynamics_path:
            raise ValueError("custom_dynamics_path is required when system_name is 'custom'")
        return CustomDynamicalSystem(custom_dynamics_path, scenario, num_inputs)

    case_path = _resolve_case_study(system_name)
    if case_path is not None:
        return CustomDynamicalSystem(str(case_path), scenario, num_inputs, name=system_name)

    fallback = _resolve_case_study("ball_beam")
    if fallback is not None:
        return CustomDynamicalSystem(str(fallback), scenario, num_inputs, name="ball_beam")

    raise ValueError(
        f"Unknown system_name={system_name!r} and no custom_dynamics_path. "
        f"Known built-ins: {sorted(set(_BUILTIN_CASE_STUDIES))}"
    )
