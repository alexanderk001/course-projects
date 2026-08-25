from collections.abc import Callable
import numpy as np
import matplotlib.pyplot as plt


def mesh_function(f: Callable[[float], float], t: np.ndarray) -> np.ndarray:
    return(np.array([f(mp) for mp in t]))


def func(t: float) -> float:
    if 0 <= t <= 3:
        return np.exp(-t)
    else:
        return np.exp(-3 * t)


def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)
    

def plot_mesh_function():
    dt = 0.1
    t = np.arange(0, 4 + dt, dt)

    y = mesh_function(func, t)

    plt.plot(t, y, 'o-', label='f(t)')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.title('Mesh function for piecewise exponential decay')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_mesh_function()
    plot_mesh_function()
