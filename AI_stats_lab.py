"""
AI_stats_lab.py

Autograded lab: Gradient Descent + Linear Regression (Diabetes)

You must implement the TODO functions below.
Do not change function names or return signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import datasets

# =========================
# Helpers (you may use these)
# =========================

def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Add a bias (intercept) column of ones to X."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return np.hstack([np.ones((X.shape[0], 1)), X])


def standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize using train statistics only.
    Returns: X_train_std, X_test_std, mean, std
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


@dataclass
class GDResult:
    theta: np.ndarray              # (d, )
    losses: np.ndarray             # (T, )
    thetas: np.ndarray             # (T, d) trajectory


# =========================
# Q1: Gradient descent + visualization data
# =========================

def gradient_descent_linreg(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 200,
    theta0: Optional[np.ndarray] = None,
) -> GDResult:
    """
    Linear regression with batch gradient descent on MSE loss.

    X should already include bias column if you want an intercept.

    Returns GDResult with final theta, per-epoch losses, and theta trajectory.
    """

    n, d = X.shape
    y = y.reshape(-1)

    if theta0 is None:
        theta = np.zeros(d)
    else:
        theta = theta0.copy()

    losses = []
    thetas = []

    for _ in range(epochs):

        y_pred = X @ theta
        error = y_pred - y

        loss = np.mean(error ** 2)

        gradient = (2 / n) * (X.T @ error)

        theta = theta - lr * gradient

        losses.append(loss)
        thetas.append(theta.copy())

    return GDResult(
        theta=theta,
        losses=np.array(losses),
        thetas=np.array(thetas)
    )


def visualize_gradient_descent(
    lr: float = 0.1,
    epochs: int = 60,
    seed: int = 0,
) -> Dict[str, np.ndarray]:

    np.random.seed(seed)

    n = 100

    X_feature = np.random.randn(n, 1)

    true_theta0 = 2.0
    true_theta1 = 3.0

    noise = np.random.randn(n) * 0.5

    y = true_theta0 + true_theta1 * X_feature[:, 0] + noise

    X = add_bias_column(X_feature)

    result = gradient_descent_linreg(
        X,
        y,
        lr=lr,
        epochs=epochs
    )

    return {
        "theta_path": result.thetas[:, :2],
        "losses": result.losses,
        "X": X,
        "y": y
    }

# =========================
# Q2: Diabetes regression using gradient descent
# =========================

def diabetes_linear_gd(
    lr: float = 0.05,
    epochs: int = 2000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:

    from sklearn.model_selection import train_test_split

    data = datasets.load_diabetes()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed
    )

    X_train, X_test, _, _ = standardize_train_test(X_train, X_test)

    X_train = add_bias_column(X_train)
    X_test = add_bias_column(X_test)

    result = gradient_descent_linreg(
        X_train,
        y_train,
        lr=lr,
        epochs=epochs
    )

    theta = result.theta

    y_train_pred = X_train @ theta
    y_test_pred = X_test @ theta

    train_mse = mse(y_train, y_train_pred)
    test_mse = mse(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q3: Diabetes regression using analytical solution
# =========================

def diabetes_linear_analytical(
    ridge_lambda: float = 1e-8,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:

    from sklearn.model_selection import train_test_split

    data = datasets.load_diabetes()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed
    )

    X_train, X_test, _, _ = standardize_train_test(X_train, X_test)

    X_train = add_bias_column(X_train)
    X_test = add_bias_column(X_test)

    d = X_train.shape[1]

    I = np.eye(d)

    theta = np.linalg.inv(
        X_train.T @ X_train + ridge_lambda * I
    ) @ X_train.T @ y_train

    y_train_pred = X_train @ theta
    y_test_pred = X_test @ theta

    train_mse = mse(y_train, y_train_pred)
    test_mse = mse(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q4: Compare GD vs analytical
# =========================

def diabetes_compare_gd_vs_analytical(
    lr: float = 0.05,
    epochs: int = 4000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Dict[str, float]:

    train_mse_gd, test_mse_gd, train_r2_gd, test_r2_gd, theta_gd = diabetes_linear_gd(
        lr,
        epochs,
        test_size,
        seed
    )

    train_mse_an, test_mse_an, train_r2_an, test_r2_an, theta_an = diabetes_linear_analytical(
        1e-8,
        test_size,
        seed
    )

    theta_l2_diff = np.linalg.norm(theta_gd - theta_an)

    theta_cosine_sim = np.dot(theta_gd, theta_an) / (
        np.linalg.norm(theta_gd) * np.linalg.norm(theta_an)
    )

    return {
        "theta_l2_diff": float(theta_l2_diff),
        "train_mse_diff": float(train_mse_gd - train_mse_an),
        "test_mse_diff": float(test_mse_gd - test_mse_an),
        "train_r2_diff": float(train_r2_gd - train_r2_an),
        "test_r2_diff": float(test_r2_gd - test_r2_an),
        "theta_cosine_sim": float(theta_cosine_sim)
    }


import matplotlib.pyplot as plt
def plot_loss_curve(losses):
    df = pd.DataFrame({
        "epoch": range(len(losses)),
        "loss": losses
    })

    df.plot(x="epoch", y="loss")
    plt.show()

def plot_theta_trajectory(theta_path):
    df = pd.DataFrame(theta_path, columns=["theta0", "theta1"])

    df.plot(x="theta0", y="theta1", marker="o")
    plt.show()

plot_loss_curve(
    losses=np.array([
        1.00,
        0.92,
        0.86,
        0.80,
        0.75,
        0.70,
        0.66,
        0.63,
        0.61,
        0.59,
        0.58
    ])
)
plot_theta_trajectory(
    theta_path=np.array([
        [0.0, 0.0],
        [0.2, -0.1],
        [0.45, -0.25],
        [0.65, -0.4],
        [0.9, -0.7],
        [1.1, -0.9],
        [1.3, -1.1],
        [1.45, -1.3],
        [1.52, -1.8]
    ])
)
