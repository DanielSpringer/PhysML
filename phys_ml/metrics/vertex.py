import numpy as np


def get_dominant_eigenvector(m: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eig(m)
    return np.abs(eigenvectors[:, np.argmax(np.abs(eigenvalues))])

