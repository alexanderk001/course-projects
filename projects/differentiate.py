import numpy as np


def differentiate(u: np.ndarray, dt: float) -> np.ndarray:
    n = len(u)
    d = np.zeros(n)
    d[0] = (u[1] - u[0]) / dt

    for i in range(1, n - 1):
        d[i] = (u[i + 1] - u[i - 1]) / (2 * dt)
    
    d[-1] = (u[-1] - u[-2]) / dt

    return d


def test_differentiate_solo():
    # exact test with a quadratic polynomial: u(t) = t**2 + 2 * t + 1
    # the exact derivative is: u'(t) = 2 * t + 2

    t = np.linspace(0, 2, 11)
    dt = t[1] - t[0]

    u = t**2 + 2 * t + 1
    expected = 2 * t + 2 * np.ones_like(t)

    computed = differentiate(u, dt)

    assert np.allclose(computed[1:-1], expected[1:-1])

 
def differentiate_vector(u: np.ndarray, dt: float) -> np.ndarray:
    d = np.zeros(len(u))

    d[0] = (u[1] - u[0]) / dt
    d[1:-1] = (u[2:] - u[0:-2]) / (2 * dt)
    d[-1] = (u[-1] - u[-2]) / dt

    return d


def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    # expected = 2 * t

    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)

    assert np.allclose(du1, du2)
    # assert np.allclose(du1, expected)
    # assert np.allclose(du2, expected)


if __name__ == '__main__':
    test_differentiate_solo()
    test_differentiate()
