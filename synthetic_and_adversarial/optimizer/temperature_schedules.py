"""
Temperature scheduling strategies for AdaSmoothES.

Temperature β controls the exploration-exploitation trade-off:
- High β (low temperature): More exploration, uniform weights
- Low β (high temperature): More exploitation, concentrated weights

This module implements various β(t) scheduling strategies.

Author: Based on simulated annealing and optimization literature
"""

import math
from abc import ABC, abstractmethod


class TemperatureSchedule(ABC):
    """Base class for temperature scheduling."""

    @abstractmethod
    def get_temperature(self, iteration: int) -> float:
        """
        Get temperature at given iteration.

        Args:
            iteration: Current iteration number (0-indexed)

        Returns:
            Temperature β_t
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name of the schedule."""
        pass


class ConstantSchedule(TemperatureSchedule):
    """
    Constant temperature: β_t = β_0

    Properties:
    - No annealing
    - Fixed exploration rate
    - Good for stationary problems
    """

    def __init__(self, beta_init: float = 10.0):
        """
        Args:
            beta_init: Constant temperature value
        """
        assert beta_init > 0, "Temperature must be positive"
        self.beta_init = beta_init

    def get_temperature(self, iteration: int) -> float:
        return self.beta_init

    def name(self) -> str:
        return f"Constant(β={self.beta_init})"


class LinearSchedule(TemperatureSchedule):
    """
    Linear decay: β_t = β_0 - decay * t

    Properties:
    - Simplest annealing
    - Linear decrease in exploration
    - Can reach β_min
    """

    def __init__(self, beta_init: float = 10.0, beta_min: float = 0.1, total_iterations: int = 10000):
        """
        Args:
            beta_init: Initial temperature
            beta_min: Minimum temperature (floor)
            total_iterations: Total number of iterations for full decay
        """
        assert beta_init > beta_min > 0, "Need beta_init > beta_min > 0"
        self.beta_init = beta_init
        self.beta_min = beta_min
        self.decay_rate = (beta_init - beta_min) / total_iterations

    def get_temperature(self, iteration: int) -> float:
        beta = self.beta_init - self.decay_rate * iteration
        return max(beta, self.beta_min)

    def name(self) -> str:
        return f"Linear(β0={self.beta_init}, min={self.beta_min})"


class ExponentialSchedule(TemperatureSchedule):
    """
    Exponential decay: β_t = β_0 * exp(-λ * t)

    Properties:
    - Fast initial decay, slow later
    - Smooth annealing curve
    - Commonly used in simulated annealing
    """

    def __init__(self, beta_init: float = 10.0, decay_rate: float = 0.001, beta_min: float = 0.01):
        """
        Args:
            beta_init: Initial temperature
            decay_rate: Decay rate λ
            beta_min: Minimum temperature (floor)
        """
        assert beta_init > 0 and decay_rate > 0, "Parameters must be positive"
        self.beta_init = beta_init
        self.decay_rate = decay_rate
        self.beta_min = beta_min

    def get_temperature(self, iteration: int) -> float:
        beta = self.beta_init * math.exp(-self.decay_rate * iteration)
        return max(beta, self.beta_min)

    def name(self) -> str:
        return f"Exponential(β0={self.beta_init}, λ={self.decay_rate})"


class PolynomialSchedule(TemperatureSchedule):
    """
    Polynomial decay: β_t = β_0 / (1 + λ * t)^p

    Properties:
    - Power p controls decay speed
    - p=1: Hyperbolic decay (default in original AdaSmoothES)
    - p>1: Faster decay
    - p<1: Slower decay
    """

    def __init__(self, beta_init: float = 10.0, decay_rate: float = 0.001, power: float = 1.0, beta_min: float = 0.01):
        """
        Args:
            beta_init: Initial temperature
            decay_rate: Decay rate λ
            power: Polynomial power p
            beta_min: Minimum temperature (floor)
        """
        assert beta_init > 0 and decay_rate > 0 and power > 0, "Parameters must be positive"
        self.beta_init = beta_init
        self.decay_rate = decay_rate
        self.power = power
        self.beta_min = beta_min

    def get_temperature(self, iteration: int) -> float:
        beta = self.beta_init / math.pow(1.0 + self.decay_rate * iteration, self.power)
        return max(beta, self.beta_min)

    def name(self) -> str:
        return f"Polynomial(β0={self.beta_init}, λ={self.decay_rate}, p={self.power})"


class CosineAnnealingSchedule(TemperatureSchedule):
    """
    Cosine annealing: β_t = β_min + 0.5 * (β_0 - β_min) * (1 + cos(π * t / T))

    Properties:
    - Smooth S-shaped curve
    - Slow at start and end, fast in middle
    - Popular in deep learning (learning rate scheduling)
    - Reaches β_min at iteration T
    """

    def __init__(self, beta_init: float = 10.0, beta_min: float = 0.1, total_iterations: int = 10000):
        """
        Args:
            beta_init: Initial temperature
            beta_min: Minimum temperature (final value)
            total_iterations: Total iterations for one annealing cycle
        """
        assert beta_init > beta_min > 0, "Need beta_init > beta_min > 0"
        self.beta_init = beta_init
        self.beta_min = beta_min
        self.total_iterations = total_iterations

    def get_temperature(self, iteration: int) -> float:
        if iteration >= self.total_iterations:
            return self.beta_min

        progress = iteration / self.total_iterations
        beta = self.beta_min + 0.5 * (self.beta_init - self.beta_min) * (1.0 + math.cos(math.pi * progress))
        return beta

    def name(self) -> str:
        return f"CosineAnnealing(β0={self.beta_init}, min={self.beta_min})"


class StepSchedule(TemperatureSchedule):
    """
    Step decay: β_t decreases by factor γ every step_size iterations

    β_t = β_0 * γ^(floor(t / step_size))

    Properties:
    - Piecewise constant
    - Sudden drops in temperature
    - Easy to tune (step_size, gamma)
    """

    def __init__(self, beta_init: float = 10.0, step_size: int = 1000, gamma: float = 0.5, beta_min: float = 0.01):
        """
        Args:
            beta_init: Initial temperature
            step_size: Number of iterations between drops
            gamma: Multiplicative decay factor (0 < γ < 1)
            beta_min: Minimum temperature (floor)
        """
        assert 0 < gamma < 1, "Gamma must be in (0, 1)"
        assert step_size > 0, "Step size must be positive"
        self.beta_init = beta_init
        self.step_size = step_size
        self.gamma = gamma
        self.beta_min = beta_min

    def get_temperature(self, iteration: int) -> float:
        num_drops = iteration // self.step_size
        beta = self.beta_init * math.pow(self.gamma, num_drops)
        return max(beta, self.beta_min)

    def name(self) -> str:
        return f"Step(β0={self.beta_init}, step={self.step_size}, γ={self.gamma})"


class AdaptiveSchedule(TemperatureSchedule):
    """
    Adaptive temperature based on optimization progress.

    Increases β (more exploration) when stuck, decreases β when improving.

    Strategy:
    - Track recent improvement rate
    - If improvement < threshold: increase β (explore more)
    - If improvement > threshold: decrease β (exploit more)
    """

    def __init__(
        self,
        beta_init: float = 10.0,
        beta_min: float = 0.1,
        beta_max: float = 100.0,
        window_size: int = 100,
        increase_factor: float = 1.1,
        decrease_factor: float = 0.95
    ):
        """
        Args:
            beta_init: Initial temperature
            beta_min: Minimum temperature
            beta_max: Maximum temperature
            window_size: Window for computing improvement rate
            increase_factor: Factor to increase β when stuck
            decrease_factor: Factor to decrease β when improving
        """
        self.beta_current = beta_init
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.window_size = window_size
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor

        # Track history
        self.loss_history = []
        self.last_update_iteration = 0

    def update(self, loss: float):
        """
        Update temperature based on observed loss.

        Call this method after each optimization step.

        Args:
            loss: Current loss value
        """
        self.loss_history.append(loss)

        # Keep only recent history
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)

    def get_temperature(self, iteration: int) -> float:
        # Update temperature periodically
        if iteration > self.last_update_iteration and iteration % 50 == 0:
            if len(self.loss_history) >= self.window_size:
                # Compute improvement rate
                recent_losses = self.loss_history[-self.window_size:]
                old_loss = sum(recent_losses[:self.window_size//2]) / (self.window_size//2)
                new_loss = sum(recent_losses[self.window_size//2:]) / (self.window_size//2)

                improvement = old_loss - new_loss

                # Adjust temperature
                if improvement < 0:  # Getting worse, increase exploration
                    self.beta_current *= self.increase_factor
                elif improvement > 0.01 * old_loss:  # Good improvement, exploit more
                    self.beta_current *= self.decrease_factor

                # Clamp
                self.beta_current = max(self.beta_min, min(self.beta_current, self.beta_max))

            self.last_update_iteration = iteration

        return self.beta_current

    def name(self) -> str:
        return f"Adaptive(β={self.beta_current:.2f})"


class CyclicSchedule(TemperatureSchedule):
    """
    Cyclic temperature: alternates between high and low temperatures.

    β_t oscillates between β_min and β_max with period T.

    Properties:
    - Periodic exploration/exploitation
    - Can escape local minima
    - Similar to cyclic learning rates
    """

    def __init__(
        self,
        beta_min: float = 1.0,
        beta_max: float = 10.0,
        cycle_length: int = 500,
        mode: str = 'triangular'
    ):
        """
        Args:
            beta_min: Minimum temperature in cycle
            beta_max: Maximum temperature in cycle
            cycle_length: Number of iterations per cycle
            mode: 'triangular', 'sine', or 'saw'
        """
        assert beta_max > beta_min > 0, "Need beta_max > beta_min > 0"
        assert cycle_length > 0, "Cycle length must be positive"
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.cycle_length = cycle_length
        self.mode = mode

    def get_temperature(self, iteration: int) -> float:
        cycle_position = (iteration % self.cycle_length) / self.cycle_length  # in [0, 1]

        if self.mode == 'triangular':
            # Triangle wave: up then down
            amplitude = 2.0 * abs(cycle_position - 0.5)  # 0 -> 1 -> 0
            beta = self.beta_min + amplitude * (self.beta_max - self.beta_min)

        elif self.mode == 'sine':
            # Sine wave
            amplitude = 0.5 * (1.0 + math.sin(2 * math.pi * cycle_position - math.pi / 2))
            beta = self.beta_min + amplitude * (self.beta_max - self.beta_min)

        elif self.mode == 'saw':
            # Sawtooth: linear up, instant drop
            beta = self.beta_min + cycle_position * (self.beta_max - self.beta_min)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return beta

    def name(self) -> str:
        return f"Cyclic(min={self.beta_min}, max={self.beta_max}, {self.mode})"


# Factory function
def get_temperature_schedule(name: str, **kwargs) -> TemperatureSchedule:
    """
    Factory function to create temperature schedule objects.

    Args:
        name: Schedule name ('constant', 'linear', 'exponential', 'polynomial',
                            'cosine', 'step', 'adaptive', 'cyclic')
        **kwargs: Additional parameters for specific schedules

    Returns:
        TemperatureSchedule object

    Examples:
        >>> schedule = get_temperature_schedule('constant', beta_init=10.0)
        >>> schedule = get_temperature_schedule('polynomial', beta_init=10.0, decay_rate=0.001, power=1.0)
        >>> schedule = get_temperature_schedule('cosine', beta_init=10.0, beta_min=0.1, total_iterations=10000)
    """
    name_lower = name.lower()

    if name_lower == 'constant':
        return ConstantSchedule(**kwargs)
    elif name_lower == 'linear':
        return LinearSchedule(**kwargs)
    elif name_lower == 'exponential':
        return ExponentialSchedule(**kwargs)
    elif name_lower == 'polynomial':
        return PolynomialSchedule(**kwargs)
    elif name_lower in ['cosine', 'cosine_annealing']:
        return CosineAnnealingSchedule(**kwargs)
    elif name_lower == 'step':
        return StepSchedule(**kwargs)
    elif name_lower == 'adaptive':
        return AdaptiveSchedule(**kwargs)
    elif name_lower == 'cyclic':
        return CyclicSchedule(**kwargs)
    else:
        raise ValueError(f"Unknown temperature schedule: {name}. "
                        f"Available: 'constant', 'linear', 'exponential', 'polynomial', "
                        f"'cosine', 'step', 'adaptive', 'cyclic'")
