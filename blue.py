"""
BLUE: Best Linear Unbiased Estimator
"""
from collections import namedtuple
import numpy as np

__version__ = "0.0.2"


class Blue(object):
    """The Blue class

    Parameters
    ----------
    data:
        A pandas dataframe containing the results and uncertainties
        per experiment. The data should be organised in a "tidy" format with
        one row per measurement. All uncertainties should be given as
        absolute uncertainties.
    correlations:
        A mapping of uncertainty
        names to correlations. Correlations can be either single numbers, in
        which case an appropriately shaped correlation matrix filled with that
        value will be created (with all diagonal elements set to one),
        a one-dimensional array-like, containing the off-diagonal elements
        of the correlation matrix, or a two-dimensional array.
    observables:
        None or a dictionary mapping observables to
        measurements. For a single observable leave as (or set to) None.

    Raises
    -------
    IndexError:
        The results column will be automatically inferred as
        the only column in the data that is not in the `correlations` mapping.
        If this can not be inferred e.g. because the number of keys in the
        mapping is not equal to the number of columns in the data minus one
        an IndexError will be raised.
    """

    _BlueResult = namedtuple(
        "BlueResult", ["weights", "covariance_matrices", "combined_covariance"]
    )

    def __init__(self, data, correlations, observables=None):
        self.data = data
        self.correlations = {i: self._to_array(j) for i, j in correlations.items()}
        self.results_column = self._get_results_col()
        self.observables = observables

    def _get_results_col(self):
        no_corr_columns = self.data.columns.symmetric_difference(self.correlations)
        if len(no_corr_columns) != 1:
            raise IndexError("Results column can not be inferred!")
        return no_corr_columns[0]

    def _run_calculation(self):
        """Run the actual BLUE calculation to find the weights.
        Also returns the calculated covariance matrices i.e.
        cov(i, j) = sigma_i * sigma_j * corr(i, j)
        """
        covariance_matrices = {}
        for i in self.data.drop(self.results_column, axis=1).columns:
            x = np.array(self.data[i].values, ndmin=2)
            covariance_matrices[i] = (x * x.T) * self.correlations[i]
        total_covariance = np.stack(list(covariance_matrices.values())).sum(axis=0)

        total_inv_covariance = np.linalg.inv(total_covariance)

        u = self._u_array(total_covariance.shape[0])

        combined_covariance = np.atleast_2d(u.T @ total_inv_covariance @ u)
        combined_covariance = np.linalg.inv(combined_covariance)

        w = (total_inv_covariance @ u)[:, np.newaxis] @ combined_covariance

        return self._BlueResult(
            weights=w.squeeze(),
            covariance_matrices=covariance_matrices,
            combined_covariance=combined_covariance,
        )

    def _u_array(self, n_measurements):
        n_obs = 1 if self.observables is None else len(self.observables)
        if n_obs > 1:
            u = np.zeros((n_measurements, n_obs))
            for i, measurements in enumerate(self.observables.values()):
                num_loc = [self.data.index.get_loc(j) for j in measurements]
                u[num_loc, i] = 1
        else:
            u = np.ones(n_measurements)
        return u

    @property
    def weights(self):
        """The BLUE weights"""
        return self._run_calculation().weights

    @property
    def _fisher_information(self):
        if self.observables is not None:
            raise NotImplementedError(
                "Information weights are only available for a single observable"
            )
        return 1 / self._run_calculation().combined_covariance.squeeze()

    @property
    def intrinsic_information_weights(self):
        """Intrinsic information weights.

        This is only available for a single observable.
        The intrinsic information weight for a measurement :math:`i` is

        .. math::
            IIW_i = \\frac{\\sigma^2_Y}{\\sigma^2_i}

        where :math:`\\sigma^2_Y` is the BLUE variance.
        See `arXiv:1307.4003 <https://arxiv.org/abs/1307.4003>`_
        for more information.

        .. note::
            The intrinsic information weights are returned for
            each input measurement. The weight attributed to correlations
            is :math:`1 - \\sum_i IIW_i`.
        """
        cov = self.total_covariance
        I = self._fisher_information

        return 1 / I / cov.diagonal()

    @property
    def marginal_information_weights(self):
        """Marginal information weights.

        This is only available for a single observable.
        The marginal information weight for a measurement :math:`i` is

        .. math::
            MIW_i = 1 - \\frac{\\sigma^2_Y}{\\sigma^2_{Y-i}}

        where :math:`\\sigma^2_Y` is the BLUE variance when including all
        measurements in the combination and :math:`\\sigma^2_{Y-i}` is the BLUE
        variance when including all measurements in the combination **except**
        measurement :math:`i`. See `arXiv:1307.4003
        <https://arxiv.org/abs/1307.4003>`_

        """
        I = self._fisher_information

        marginal_weights = [
            (I - self[self.data.index.drop(i)]._fisher_information) / I
            for i in self.data.index
        ]

        return np.array(marginal_weights)

    @property
    def combined_result(self):
        """The combined result.

        A single number if only a single observable, a dictionary of
        observables and results if more than one observable.
        """
        w = self.weights
        if w.ndim == 1:
            return (self.data[self.results_column] * w).sum()

        return dict(zip(self.observables, self.data[self.results_column] @ w))

    @property
    def combined_uncertainties(self):
        """The uncertainties on the combined result(s)"""
        result = self._run_calculation()
        w, covs = result.weights, result.covariance_matrices
        uncerts = {}
        for i, j in covs.items():
            weighted_cov = w.T @ j @ w
            if weighted_cov.ndim:
                weighted_cov = weighted_cov.diagonal()
            uncerts[i] = np.sqrt(weighted_cov)
        return uncerts

    @property
    def pulls(self):
        """Get the pulls"""
        if self.observables is not None:
            raise NotImplementedError("Pulls only valid for a single observable")
        diff_result = self.data[self.results_column] - self.combined_result
        diff_variance = np.sqrt(
            self.total_covariance.diagonal() - (1 / self._fisher_information)
        )
        return diff_result / diff_variance

    @property
    def total_covariance(self):
        """The total covariance matrix.

        This is constructed from the input data and correlations passed into
        the class constructor. For each source of uncertainty, a covariance
        matrix is constructed such that :math:`Cov(i, j) = \\sigma_i \\sigma_j
        \\rho_{ij}`. The total covariance is then obtained by adding all the
        covariance matrices element wise.
        """
        covs = self._run_calculation().covariance_matrices
        return np.stack(list(covs.values())).sum(axis=0)

    @property
    def total_correlations(self):
        """The total correlation matrix"""
        total_covariance = self.total_covariance
        sigmas = np.sqrt(np.diagonal(total_covariance)).reshape(-1, 1)
        return total_covariance / (sigmas * sigmas.T)

    @property
    def combined_covariance(self):
        """The covariance matrix of the combination.

        In the case of a single observable this will just be a single number
        but is always returned as a two-dimensional numpy array.
        """
        return self._run_calculation().combined_covariance

    @property
    def observable_correlations(self):
        """The correlation between multiple observables.

        This property is only really useful when performing the blue
        combination with multiple observables as the correlation with an
        observable with itself is just one.
        """
        comb_cov = self._run_calculation().combined_covariance
        diag = np.atleast_2d(comb_cov.diagonal())
        return comb_cov / np.sqrt(diag * diag.T)

    @property
    def chi2_ndf(self):
        """The :math:`\\chi^2` and number-of-degrees-of-freedom (NDF) of the
        combination.

        One can obtain the p-value of the combination using scipy::

            from scipy.stats import chi2

            blue = Blue(data, correlations)
            p_value = chi2.sf(*blue.chi2_ndf)
        """
        diff = self._measured_diff()
        nobs = len(self.observables) if self.observables else 1
        ndf = len(self.data) - nobs
        return diff.T @ np.linalg.inv(self.total_covariance) @ diff, ndf

    def _measured_diff(self):
        """The difference between the combined result and inputs"""
        combined_result = self.combined_result
        diff = self.data[self.results_column].copy()

        if not self.observables:
            return diff - combined_result

        for i, j in self.observables.items():
            diff[j] -= combined_result[i]
        return diff

    def __getitem__(self, item):
        """Make a new instance of the :py:class:`Blue` class using only a
        subset of measurements.

        This makes it easy to perform a combination using a
        subset of all the measurements without having to redefine correlation
        matrices.

        Parameters
        ----------
        item : list
            A list of measurements

        Returns
        -------
        :class:`Blue`:
            A new instance of the :py:class:`Blue` class
            using a subset of measurements as defined by `item`.

        """
        num_loc = [self.data.index.get_loc(i) for i in item]
        sub_df = self.data.loc[item]

        new_correlations = {
            i: j[num_loc, :][:, num_loc] for i, j in self.correlations.items()
        }

        return Blue(sub_df, new_correlations)

    def _to_array(self, in_array):
        in_array = np.array(in_array)
        out_len = len(self.data)
        out_shape = (out_len, out_len)

        if in_array.ndim == 0:
            out = np.full(out_shape, in_array)
            np.fill_diagonal(out, 1.0)
        elif in_array.ndim == 1:
            tri_indices = np.triu_indices(out_len, k=1)
            if len(in_array) == len(tri_indices[0]):
                out = np.ones(out_shape)
                out[tri_indices] = in_array
                out.T[tri_indices] = in_array
            else:
                raise ValueError(
                    "One dim. correlations supplied which are taken to be the "
                    "upper triangle elements of the correlations matrix. "
                    f"There should be {len(tri_indices[0])} elements, "
                    f"you have supplied {len(in_array)}"
                )
        elif in_array.ndim == 2:
            if in_array.shape == out_shape:
                out = in_array
            else:
                raise ValueError(
                    "Correlation matrix is not the correct shape. "
                    f"(should be {out_shape}, is {in_array.shape})"
                )
        else:
            raise ValueError(
                f"{in_array.ndim}D correlation matrix doesn't make sense. "
                "Input correlations can be 0, 1 or 2D."
            )

        return out

    @classmethod
    def iterative(cls, data, correlations, fixed=None, cutoff=0.01, max_iters=200):
        """Construct an instance of the Blue class iteratively,
        updating uncertainties
        based on the combined result and repeating the combination until the
        change between successive iterations is less that cutoff * 100 %. The
        first two parameters of this method are the same as those used to
        initialise the standard :py:class:`Blue` class.

        Parameters
        ----------
        data:
            See :py:class:`Blue`
        correlations:
            See :py:class:`Blue`
        fixed:
            Uncertainty names that should not be scaled when
            applying the iterative procedure.
            These are typically statistical uncertainties.
        cutoff: float
            The point at which to stop iterating and return.
        max_iters: int
            When to stop iterating and give up.

        Returns
        -------
        :class:`Blue`
            An instance of the Blue class constructed iteratively

        Raises
        ------
        RuntimeError:
            if `max_iters` is reached.

        """
        it_data = data.copy()
        prev_result = None

        for _ in range(max_iters):
            blue = cls(it_data, correlations)
            result = blue.combined_result
            if prev_result and abs(1 - result / prev_result) < cutoff:
                break
            prev_result = result
            new_uncerts = (
                (
                    data.drop(blue.results_column, axis=1).T
                    / it_data[blue.results_column].values
                )
                * result
            ).T
            if fixed is not None:
                new_uncerts = new_uncerts.drop(fixed, axis=1)
            it_data.update(new_uncerts)

        else:
            raise RuntimeError(f"Failed to converge after {max_iters} iterations!")
        return blue
