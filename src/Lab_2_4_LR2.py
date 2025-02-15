import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        # convierte a matriz la X si no lo era -> haremos fit multiple no fit simple. 
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        # esta línea añade una fila de unos a la matriz X, por lo que al fittear tenemos el b. 
        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Replace this code with the code you did in the previous laboratory session

        # Store the intercept and the coefficients of the model

        #recuperando el código de la práctica 3: 
        # sabemos, por la teoria, que 
        # y = bo + b1 x1 + b2 x2 + ... + bn xn 
        # w = (Xt * X)^(-1) * Xt * y 
        # donde bo será el primer término de w 
        X_completa = X

        # no tenemos que insertar una fila de 1s porque ya lo inserta el fit. 
        # dejamos esta parte comentada. 
        # fila_unos = np.ones((X.shape[0], 1)) 
        # X_completa = np.hstack((fila_unos, X))
        
        #y_completa = np.append(1, y)
        #calculamso el verctor w = (bo, b1, b2, ...)

        Xt_X_inv =  np.linalg.inv(np.dot(np.transpose(X_completa), X_completa))
        Xt_X_inv_Xt = np.dot (Xt_X_inv , np.transpose(X_completa))
        resultados  = np.dot (Xt_X_inv_Xt, y)
        #tomamos los coeficientes (b1, .., bn)
        self.coefficients = resultados[1:]
        # tomamos el bo como el intercept 
        self.intercept = resultados[0]


    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        # Initialize the parameters to very small values (close to 0)
        m = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01

        # Implement gradient descent (TODO)


        for epoch in range(iterations):
            # vemos , para nuestros valores 
            # predictions = self.fit(self.coefficients, y)
            predictions =  np.dot(X[:, 1:], self.coefficients) + self.intercept 
            error = predictions - y

            # TODO: Write the gradient values and the updates for the paramenters

            gradient_b = np.sum(error) / m
            gradient_w =   np.dot(X[:, 1:].T, error) / m

            self.intercept -= learning_rate * gradient_b
            self.coefficients -= learning_rate * gradient_w 

            # TODO: Calculate and print the loss every 10 epochs
            if epoch % 1000 == 0:
                mse = np.sum(error**2) / m  
                print(f"Epoch {epoch}: MSE = {mse}")

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week
    

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        X = np.asarray(X)
        w = self.coefficients
        b = self.intercept
        if np.ndim(X) == 1:
            # TODO: Predict when X is only one variable
            # y = wx + b 

            # x lo convertimos en un array de 2D con una columna 
            X = X.reshape(-1, 1)

            # si w es un solo numero, lo convertios en un array: 
            if w.ndim == 0:
                w = np.array([w])

        else: 
       
            # TODO: Predict when X is more than one variable 
            # calculamos y = b + x * w cuando X es una matriz 2D 
            # convertimos la w en una matriz para poder hacer el calculo. 

            w = np.asarray(w)
            if w.ndim == 1:
                w = w.reshape(-1, 1)
            
        predictions =  b + X @ w

        return predictions.flatten()

        
    


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    # R^2 Score
    # TODO
    RSS = np.sum((y_true-y_pred)**2)
    TSS = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (RSS/TSS)

    # Root Mean Squared Error
    # TODO
    N = len(y_true)
    rmse = np.sqrt(np.sum((y_true - y_pred)**2)/N) 


    # Mean Absolute Error
    # TODO
    mae = np.sum(abs(y_true - y_pred))/N

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()
    for index in sorted(categorical_indices, reverse=True):
        # TODO: Extract the categorical column
        categorical_column = None

        # TODO: Find the unique categories (works with strings)
        unique_values = None


        # TODO: Create a one-hot encoded matrix (np.array) for the current categorical column
        one_hot = None

        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]

        # TODO: Delete the original categorical column from X_transformed and insert new one-hot encoded columns
        X_transformed = None

    return X_transformed
