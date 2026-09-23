import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        #We do not want to modify the initial_weights direcrtly.
        weights = np.array(initial_weights, dtype =np.float64)
        #number of traiing examples
        N= len(X)

        for _ in range(num_iterations):
            # Compute predictions with get_model_prediction(X, weights)
            predictions = self.get_model_prediction(X, weights)

            #compute gradient for every step
            gradients =np.zeros_like(weights)
            
            for j in range(len(weights)):
                gradients[j] = self.get_derivative(
                    predictions,Y, N, X, j
                )
            #Update Gradient descent update
            weights -=  self.learning_rate * gradients

        return np.round(weights, 5)
