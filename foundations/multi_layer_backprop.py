import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        x = np.array(x)
        W1 =np.array(W1)
        b1 = np.array(b1)
        W2= np.array(W2)
        b2 = np.array(b2)

        y_true =np.array(y_true)

    

        #----- FORDWARD  PASS -----

        #First Linear layer
        z1= W1 @ x + b1 
        # RELU
        a1 = np.maximum(0.0, z1)

        #Second Linear Layer 
        y_hat = W2 @ a1 +b2

        #mean square error - loss funciton
        mse_loss = np.mean((y_hat -  y_true)**2)


        #----- BACKWARD PASS (Back prop) -----
        
        # gradient  of mse w.r.t y_hat dl/dz2
        dz2= (2.0/y_true.size) * (y_hat - y_true)

        #gradient of second layer dl/sW2
        dW2 = np.outer(dz2, a1)
        db2 = dz2 

        # Send gradient backward through W2
        da1=W2.T @ dz2

        #Relu derivative
        dz1 = da1 * (z1>0).astype(float)

        # Gradients of first layer
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            "loss": round(float(mse_loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist()
        }
