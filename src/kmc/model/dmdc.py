from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.exceptions import NotFittedError
import numpy as np
from numpy.typing import ArrayLike

class DMDc(BaseEstimator):
    """
    Dynamic Mode Decomposition with Control (DMDc).
    
    Fits a discrete-time linear model: x_{k+1} = A * x_k + B * u_k
    
    Regression Methods:
        - 'ols': Ordinary Least Squares. Minimizes ||Error||^2.
        - 'ridge': L2 Regularization. Minimizes ||Error||^2 + alpha * ||W||^2.
        - 'lasso': L1 Regularization. Minimizes ||Error||^2 + alpha * |W|.
    """

    def __init__(self, method: str = 'ridge', alpha: float = 1.0):
        """
        Args:
            method: The regression formulation to use. Options: ['ols', 'ridge', 'lasso'].
            alpha:  Regularization strength (used for 'ridge' and 'lasso'). 
                    Ignored if method is 'ols'.
        """
        self.method = method
        self.alpha = alpha
        
        # Learned System Matrices
        self.A = None
        self.B = None
        
        # Metadata
        self.n_features_in_ = None 

    def fit(self, X1: ArrayLike, X2: ArrayLike, U: ArrayLike) -> 'DMDc':
        """
        Fit A and B matrices using the specified regression method.
        """
        X_current = np.asarray(X1)
        X_next = np.asarray(X2)
        U_current = np.asarray(U)
        
        self.n_features_in_ = X_current.shape[1]

        # Construct Omega (Feature Matrix)
        Omega = np.concatenate([X_current, U_current], axis=1)  # Shape: (n_samples, n_states + n_controls)
        
        # --- Select Regression Method ---
        if self.method == 'ridge':
            # Ridge uses solvers like 'auto', 'svd', 'cholesky' internally
            regressor = Ridge(alpha=self.alpha, fit_intercept=False)
            
        elif self.method == 'ols':
            # OLS uses 'svd' (singular value decomposition) internally
            regressor = LinearRegression(fit_intercept=False)
            
        elif self.method == 'lasso':
            # Lasso uses 'coordinate_descent' solver internally
            regressor = Lasso(alpha=self.alpha, fit_intercept=False)
            
        else:
            raise ValueError(f"Unknown method: {self.method}. Supported: ['ols', 'ridge', 'lasso']")

        # Solve for coefficients
        regressor.fit(Omega, X_next)
        
        # Extract Matrices from coefficients
        K = regressor.coef_ # Shape: (n_states, n_states + n_controls)
        
        n_states = X_current.shape[1]
        self.A = K[:, :n_states]
        self.B = K[:, n_states:]
        
        return self

    def predict(self, X: ArrayLike, U: ArrayLike) -> np.ndarray:
        """
        Predict next state: x_{k+1} = A x_k + B u_k
        """
        if self.A is None or self.B is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet.")
            
        X = np.asarray(X)
        U = np.asarray(U)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Feature mismatch: Expected {self.n_features_in_}, got {X.shape[1]}")

        # Compute using Transpose (A.T) because input vectors are rows
        return X @ self.A.T + U @ self.B.T