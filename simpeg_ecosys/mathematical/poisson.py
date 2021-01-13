"""
The implementation of the Poisson equation using the finite volume method.
The code is referenced from `simpeg` and based on the `discretize`.
"""
import warnings

import numpy as np
import scipy.sparse as sp
from discretize.base import BaseMesh
from discretize.utils import closest_points_index, is_scalar, sdiag

from ..utils import all_equal, flatten, getxBCyBC_CC

ITERABLE_CONTAINER = (list, tuple, np.ndarray)


# TODO: check eval method
class VolumeSource(object):
    """The source is the uniform value of the entire control volume."""

    def __init__(self, locations, values=None):
        """Define the location and the value of the sources.

        Parameters
        ----------
        locations : list, tuple or numpy.ndarray
            The location of the source term.
        values : list, tuple or numpy.ndarray, optional
            The value of the source term.
        """
        _locations = self._assign_locations(locations)
        _values = self._assign_values(_locations, values)
        unique_locations, indices = np.unique(
            _locations,
            axis=0, return_index=True
        )
        if _locations.shape[0] != unique_locations.shape[0]:
            warnings.warn('Duplicated location!')
        self._locations = unique_locations
        self._values = _values[indices]

    @ property
    def locations(self):
        return self._locations

    @ property
    def values(self):
        return self._values

    def eval(self, simulator):
        warnings.warn("Correctness of 'f' not yet confirmed.")
        source_location_indices = closest_points_index(
            simulator.mesh, self._locations)
        self._f = np.zeros(simulator.mesh.n_cells)
        for source_location_index, source_value in \
                zip(source_location_indices, self._values):
            # I don’t know why we can do this, and I’m not sure if it’s correct
            self._f[source_location_index] += (
                source_value * simulator.mesh.vol[source_location_index]
            )
        return self._f

    def add_source(self, locations, values):
        _locations = self._assign_locations(locations)
        _values = self._assign_values(_locations, values)

        _num_locations = self._locations.shape[0] + _locations.shape[0]
        # unique locations and indices
        self._locations, indices = np.unique(
            np.vstack((self._locations, _locations)),
            axis=0, return_index=True
        )

        # number of unique locations not equal number of total locations
        if self._locations.shape[0] != _num_locations:
            print(self._locations.shape[0], _num_locations)
            warnings.warn('Duplicated location!')
        self._values = np.hstack(
            (self._values, _values)
        )[indices]

    def _assign_locations(self, locations):
        # Sanity Checks
        if not isinstance(locations, ITERABLE_CONTAINER):
            raise TypeError(
                f"locations must be a list, tuple or np.ndarray, not {type(locations)}"
            )
        elif not self._legal_locations(locations):
            raise TypeError(
                "locations should be a 2D container form with scaler elements."
                " And the second dimension should between 1 to 3."
            )
        else:
            return np.array(locations, dtype=np.float64)

    def _assign_values(self, locations, values):
        if values is None:
            return np.ones(self.locations.shape[0])
        elif not isinstance(values, ITERABLE_CONTAINER):
            raise TypeError(
                f"values must be a list, tuple or np.ndarray, not {type(locations)}"
            )
        elif not self._legal_values(values):
            raise TypeError(
                "values should be a 1D container form with scaler elements."
            )
        else:
            return np.array(values, dtype=np.float64)

    def _legal_locations(self, container):
        elem_length = []
        for elem in container:
            if isinstance(elem, ITERABLE_CONTAINER):
                elem_length.append(len(elem))
                if not all(map(is_scalar, elem)):
                    return False
            else:
                return False
        if all_equal(elem_length) \
                and 1 <= elem_length[0] <= 3:
            return True
        else:
            return False

    def _legal_values(self, values):
        if all(map(is_scalar, values)):
            return True
        else:
            return False

    def __repr__(self):
        with np.printoptions(precision=4):
            num_elems = self._locations.shape[1]
            width1 = num_elems * 10 + num_elems + 1
            width2 = 10
            fmt = f"\n    {'locations':>{width1}}    "
            fmt += "|    " + f"{'values':>{width2}}\n"
            fmt += (width1 + 8) * "-" + "+"
            fmt += (width2 + 8) * "-" + "\n"
            for i in range(self._locations.shape[0]):
                fmt += "    "
                fmt += np.array2string(
                    self._locations[i],
                    formatter={'float_kind': lambda x: "%10.4f" % x}
                )
                fmt += "    |    "
                fmt += np.array2string(
                    self._values[i],
                    formatter={'float_kind': lambda x: "%10.4f" % x}
                )
                fmt += "    \n"
        return fmt


# TODO: check eval method
class PointSource(VolumeSource):
    """The source is the unit value in the control volume."""

    def __init__(self, locations, values):
        """Define the location and the value of the sources.

        Parameters
        ----------
        locations : list, tuple or numpy.ndarray
            The location of the source term.
        values : list, tuple or numpy.ndarray, optional
            The value of the source term.
        """
        super().__init__(locations, values=values)

    def eval(self, simulator):
        warnings.warn("Correctness of 'f' not yet confirmed.")
        source_location_indices = closest_points_index(
            simulator.mesh, self._locations)
        self._f = np.zeros(simulator.mesh.n_cells)
        for source_location_index, source_value in \
                zip(source_location_indices, self._values):
            # I don’t know why we can do this, and I’m not sure if it’s correct
            self._f[source_location_index] += source_value
        return self._f


# TODO: check RHS_BC
class PoissonCellCentered(object):
    """Cell centered Poisson problem
    .. math::
        \\nabla \\cdot \\sigma (- \\nabla \\phi) = f
    """

    def __init__(self, mesh, bc_types,
                 bc_values=None, source_list=None, model_parameters=None,
                 **kwargs):
        """
        Define Poisson's equation with mesh, boundary conditions,
        diffusion coefficient and source terms.

        Parameters
        ----------
        mesh : discretize.base.BaseMesh
            The mesh to discretize the poisson problem.
        bc_types : list of lists of str
            'dirichlet' or 'neumann' boundary condition.
        bc_values : list of lists of float, optional
            The value of each boundary condition.
            By default, None means all zeros.
        source_list : list of VolumeSource (PointSource), optional
            Source or sink, by default None.
        model_parameters : numpy.ndarray, optional
            Material property (tensor properties are possible)
            at each cell center (n_cells, (1, 3, or 6)), by default None.
        """
        if not isinstance(mesh, BaseMesh):
            raise(TypeError(
                f"mesh must be an instance of discretize.base.BaseMesh, not {type(mesh)}"
            ))
        if not isinstance(source_list, (list, type(None))):
            raise(TypeError(
                f"source_list must be an instance of list, not {type(source_list)}"
            ))

        if "verbose" in kwargs:
            self.verbose = kwargs["verbose"]
        else:
            self.verbose = False
        self.mesh = mesh
        self.MfParams = self.mesh.get_face_inner_product(
            model_parameters, invert_model=True, invert_matrix=True)
        self.source_list = source_list
        self.set_BC(bc_types, bc_values)

    def getA(self, model_parameters=None, **kwarg):
        """
        Make the A matrix for the cell centered Poisson's problem
        A = D MfParams G
        """
        D = self.Div
        G = self.Grad
        if model_parameters is None:
            MfParams = self.MfParams
        else:
            MfParams = self.mesh.get_face_inner_product(
                model_parameters, invert_model=True, invert_matrix=True)
        A = D @ MfParams @ G

        if 'dirichlet' not in list(flatten(self.mesh._cell_gradient_BC_list)):
            if self.verbose:
                print(
                    "Perturbing first row of A to remove nullspace for Neumann BC."
                )
            # Handling Null space of A
            _, J, _ = sp.sparse.find(A[0, :])
            for jj in J:
                A[0, jj] = 0.0
            A[0, 0] = 1.0

        return A

    def getRHS(self):
        """RHS for the Poisson's equation."""
        if self.source_list is not None:
            RHS = np.tile(self.RHS_BC[:, np.newaxis], len(self.source_list))
            for i, source in enumerate(self.source_list):
                RHS[:, i] += source.eval(self)
            if RHS.shape[1] == 1:
                RHS = RHS.flatten()
        else:
            RHS = self.RHS_BC.flatten()

        return RHS

    def set_BC(self, bc_types, bc_values=None):
        """Set boundary conditions."""
        self.mesh.set_cell_gradient_BC(bc_types)

        if self.mesh.dim == 1:
            # fxm: boundary faces at x-start (minus)
            # fxp: boundary faces at x-end (positive)
            fxm, fxp = self.mesh.face_boundary_indices
            gBF = (self.mesh.faces_x[fxm], self.mesh.faces_x[fxp])
            if bc_values is None:
                bc_values = [[0, 0]]
        elif self.mesh.dim == 2:
            # fym: boundary faces at y-start (minus)
            # fyp: boundary faces at y-end (positive)
            fxm, fxp, fym, fyp = self.mesh.face_boundary_indices
            gBF = (self.mesh.faces_x[fxm, :], self.mesh.faces_x[fxp, :],
                   self.mesh.faces_y[fym, :], self.mesh.faces_y[fyp, :])
            if bc_values is None:
                bc_values = [[0, 0], [0, 0]]
        else:  # self.mesh.dim == 3
            fxm, fxp, fym, fyp, fzm, fzp = self.mesh.face_boundary_indices
            gBF = (self.mesh.faces_x[fxm, :], self.mesh.faces_x[fxp, :],
                   self.mesh.faces_y[fym, :], self.mesh.faces_y[fyp, :],
                   self.mesh.faces_z[fzm, :], self.mesh.faces_z[fzp, :])
            if bc_values is None:
                bc_values = [[0, 0], [0, 0], [0, 0]]

        # Setup Mixed B.C (alpha, beta, gamma)
        alpha = []
        beta = []
        gamma = []
        for j in range(self.mesh.dim):
            for i in range(2):
                if j == 0 and self.mesh.dim == 1:
                    temp = np.ones_like(gBF[j * 2 + i][j])
                else:
                    temp = np.ones_like(gBF[j * 2 + i][:, j])
                if self.mesh._cell_gradient_BC_list[j][i] == "dirichlet":
                    alpha_temp = temp
                    beta_temp = temp * 0.0
                    gamma_temp = temp * bc_values[j][i]
                elif self.mesh._cell_gradient_BC_list[j][i] == "neumann":
                    alpha_temp = temp * 0.0
                    beta_temp = temp
                    gamma_temp = temp * bc_values[j][i]
                else:
                    raise NotImplementedError(
                        "Only implement 'dirichlet' and 'neumann' BC."
                    )
                alpha.append(alpha_temp)
                beta.append(beta_temp)
                gamma.append(gamma_temp)

        self.x_BC, self.y_BC = getxBCyBC_CC(self.mesh, alpha, beta, gamma)
        self.V = sdiag(self.mesh.vol)
        self.Div = self.V * self.mesh.face_divergence
        self.P_BC, self.B = self.mesh.get_BC_projections_simple()
        self.M = self.B * self.mesh.average_cell_to_face
        self.Grad = self.Div.T - self.P_BC * sdiag(self.y_BC) * self.M
        warnings.warn("Correctness of 'RHS_BC' not yet confirmed.")
        # I don’t know why we can do this, and I’m not sure if it’s correct
        self.RHS_BC = np.diag(
            2 *
            self.Div
            @ self.P_BC @ sdiag(abs(self.x_BC)) @ self.M
            / self.mesh.vol
        )
