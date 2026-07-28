import numpy as np
import sympy as sp
from poisson import Poisson
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

x, y = sp.symbols("x,y")

# Below we create a solver that reuses some of the implementation from
# the 1D solver in poisson.py.


class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with Dirichlet boundary conditions.
    """

    def __init__(self, Lx: float, Ly: float):
        self.px = Poisson(Lx)  # we can reuse some of the code from the 1D case
        self.py = Poisson(Ly)

    def create_mesh(self, Nx: int, Ny: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a 2D Cartesian mesh

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in x-direction
        Ny : int
            The number of uniform intervals in y-direction
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh
        """
        raise NotImplementedError

    def laplace(self, Nx: int, Ny: int) -> sparse.lil_matrix:
        """Return a vectorized Laplace operator

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in x-direction
        Ny : int
            The number of uniform intervals in y-direction

        Returns
        -------
        A : scipy sparse LIL matrix
            The vectorized Laplace operator
        """
        raise NotImplementedError

    def assemble(self, Nx: int, Ny: int, f: sp.Expr, ue: sp.Expr) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return assembled coefficient matrix A and right hand side vector b

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in x-direction
        Ny : int
            The number of uniform intervals in y-direction
        f : Sympy expression
            The right hand side as a Sympy expression in x and y
        ue : Sympy expression
            The exact solution as a Sympy expression in x and y

        Returns
        -------
        A : scipy sparse CSR matrix
            Coefficient matrix
        b : 1D array
            Right hand side vector
        """
        raise NotImplementedError

    def l2_error(self, u: np.ndarray, ue: sp.Expr) -> float:
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy expression
            The exact solution

        Returns
        -------
        float - The l2-error
        """
        raise NotImplementedError

    def __call__(self, Nx: int, Ny: int, ue: sp.Expr) -> np.ndarray:
        """Solve Poisson's equation with a given manufactured solution

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in x-direction
        Ny : int
            The number of uniform intervals in y-direction
        ue : Sympy expression
            The exact solution

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(Nx, Ny, sp.diff(ue, x, 2) + sp.diff(ue, y, 2), ue)
        return sparse_linalg.spsolve(A, b.ravel()).reshape((Nx + 1, Ny + 1))


def test_poisson2d():
    return False # TODO: implement a test for the 2D Poisson solver
