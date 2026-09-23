import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        #predicted value
        z= np.dot(x,w) +b
        #activation function - sigmoid prediction
        y_hat = 1/(1+np.exp(-z))
        gradient_dw = (y_hat - y_true) * (y_hat*(1-y_hat))*x

        gradient_db =  (y_hat - y_true) *(y_hat*(1-y_hat))

        return np.round(gradient_dw, 5), round(float(gradient_db), 5)


