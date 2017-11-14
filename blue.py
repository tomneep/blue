"""
TODO:
    - Compatibiltiy
    - Parameters
    - Pulls
"""

import numpy as np


class Blue(object):

    def __init__(self, data, correlations, results_column=None,
                 observables=None):
        self.data = data
        self.correlations = {i: self._to_array(j)
                             for i, j in correlations.items()}
        self.results_column = results_column or self._get_results_col()
        self.observables = observables

    def _get_results_col(self):
        no_corr_columns = self.data.columns ^ self.correlations
        if len(no_corr_columns) != 1:
            raise IndexError('Results column can not be infered!')
        return no_corr_columns[0]

    def _run_calculation(self):
        covariance_matrices = {}
        for i in self.data.drop(self.results_column, axis=1).columns:
            x = np.array(self.data[i].values, ndmin=2)
            covariance_matrices[i] = (x * x.T) * self.correlations[i]
        total_covariance = np.stack(covariance_matrices.values()).sum(axis=0)

        total_inv_covariance = np.linalg.inv(total_covariance)

        if self.observables is None:
            u = np.ones(total_inv_covariance.shape[0])
        else:
            u = self._get_u_array(total_inv_covariance.shape[0])

        covariance_result = u.T @ total_inv_covariance @ u

        if self.observables is None:
            w = (total_inv_covariance @ u) / covariance_result
        else:
            covariance_result = np.linalg.inv(covariance_result)
            w = (total_inv_covariance @ u) @ covariance_result

        return w, covariance_matrices

    def _get_u_array(self, n_measurements):
        n_obs = len(self.observables)
        u = np.zeros((n_measurements, n_obs))
        for i, measurements in enumerate(self.observables.values()):
            num_loc = [self.data.index.get_loc(j) for j in measurements]
            u[num_loc, i] = 1
        return u

    @property
    def weights(self):
        w, _ = self._run_calculation()
        return w

    @property
    def _fisher_information(self):
        if self.observables is not None:
            raise NotImplementedError(
                'Information weights are only available '
                'for a single observable'
            )
        cov = self.total_covariance
        u = np.ones(cov.shape[0])
        return u.T @ np.linalg.inv(cov) @ u

    @property
    def intrinsic_information_weights(self):
        """See arXiv:1307.4003"""
        cov = self.total_covariance
        I = self._fisher_information

        return 1 / I / cov.diagonal()

    @property
    def marginal_information_weights(self):
        """See arXiv:1307.4003"""
        I = self._fisher_information

        marginal_weights = [
            (I - self[self.data.index.drop(i)]._fisher_information) / I
            for i in self.data.index
        ]

        return np.array(marginal_weights)

    @property
    def combined_result(self):
        w = self.weights
        if w.ndim == 1:
            return (self.data[self.results_column] * w).sum()

        return dict(zip(self.observables, self.data[self.results_column] @ w))

    @property
    def combined_uncertainties(self):
        w, covs = self._run_calculation()
        return {i: np.sqrt(w.T @ j @ w) for i, j in covs.items()}

    @property
    def total_covariance(self):
        _, covs = self._run_calculation()
        return np.stack(covs.values()).sum(axis=0)

    @property
    def total_correlations(self):
        total_covariance = self.total_covariance
        sigmas = np.sqrt(np.diagonal(total_covariance)).reshape(-1, 1)
        return total_covariance / (sigmas * sigmas.T)

    @property
    def chi2_ndf(self):
        combined_result = self.combined_result
        total_covariance = self.total_covariance
        diff = self.data[self.results_column].values - combined_result
        ndf = len(self.data) - 1
        return diff.T @ np.linalg.inv(total_covariance) @ diff, ndf

    def __getitem__(self, item):
        num_loc = [self.data.index.get_loc(i) for i in item]
        sub_df = self.data.loc[item]

        new_correlations = {i: j[num_loc, :][:, num_loc]
                            for i, j in self.correlations.items()}

        return Blue(sub_df, new_correlations, self.results_column)

    def _to_array(self, in_array):
        out = np.array(in_array)
        out_len = len(self.data)
        out_shape = (out_len, out_len)
        if out.ndim == 0:
            out = np.full(out_shape, out)
            np.fill_diagonal(out, 1.0)
        if out.shape != out_shape:
            raise IndexError(
                'Correlation matrix is not the correct shape. '
                f'(should be {out_shape}, is {out.shape})')
        return out

    @classmethod
    def iterative(cls, data, correlations, results_column=None,
                  fixed=None, cutoff=0.01, max_iters=200):
        it_data = data.copy()
        prev_result = None

        for _ in range(max_iters):
            blue = cls(it_data, correlations, results_column=results_column)
            result = blue.combined_result
            if prev_result and abs(1 - result / prev_result) < cutoff:
                break
            prev_result = result
            new_uncerts = (
                (data.drop(blue.results_column, axis=1).T
                 / it_data.Value.values) * result
            ).T
            if fixed is not None:
                new_uncerts = new_uncerts.drop(fixed, axis=1)
            it_data.update(new_uncerts)

        else:
            raise RuntimeError(
                f'Failed to converge after {max_iters} iterations!'
            )
        return blue
