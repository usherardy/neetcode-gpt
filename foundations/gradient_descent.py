class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:

        x = init
        # Objective function: f(x) = x^2
        for _ in range(iterations):
            gradient = 2*x
        # Derivative:         f'(x) = 2x
            x = x - learning_rate * gradient
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        return round(x, 5)
