Blue: Python implementation of BLUE - Best Linear Unbiased Estimator
====================================================================

Blue exists as a single python file, containing a single class,
:py:class:`Blue`. :py:class:`Blue` uses the power of :py:mod:`pandas` and
:py:mod:`numpy` leading to a relatively simple implementation.
:py:mod:`blue` requires Python 3.6 due its use of two particularly delicious
language features: the matrix multiplication operator `@` and f-strings.
:py:mod:`blue` doesn't use f-strings extensively and so could be made to work
with Python 3.5, but hey, why use 3.5 when you can use 3.6!

Using :py:mod:`blue` to get a combined result is straight-forward, the
difficult part is making you measurements and assigning correlations between the
uncertainties of those measurements. Once this has been achieved one can
construct an instance of the :py:class:`Blue` class to find the combined result
and inspect its properties. The available methods to perform the combination are
documented below but :py:mod:`blue` is best demonstrated by example.

.. toctree::
   :maxdepth: 1
   :caption: Examples

   notebooks/ATLAS-CONF-2013-098
   notebooks/ATLAS-CONF-2014-054
   notebooks/arxiv_1709.05327
   notebooks/ATLAS-CONF-2013-102

.. todo::
   Add notebooks for more combinations


The Blue class
==============

.. automodule:: blue

.. autoclass:: Blue

   .. automethod:: iterative
   .. automethod:: __getitem__

   The combined result and its uncertainties can be obtained with

   .. autoattribute:: combined_result
   .. autoattribute:: combined_uncertainties

   Several different weights can be inspected as properties of the class

   .. autoattribute:: weights
   .. autoattribute:: intrinsic_information_weights
   .. autoattribute:: marginal_information_weights

   The total covariance and correlation matrices can be accessed with the properties:

   .. autoattribute:: total_covariance
   .. autoattribute:: total_correlations

   .. autoattribute:: observable_correlations

   The consistency of the combination can be tested using:

   .. autoattribute:: chi2_ndf
   .. autoattribute:: pulls

..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
