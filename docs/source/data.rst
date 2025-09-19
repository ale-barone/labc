Data
====

.. automodule:: labc.data
   :no-members:

The submodules **data** provides the interface to handle data. Once a ``StatsType`` object is initialized,
we can store the resampled bins in a ``DataStats`` objects and perform all the mathematical operation
directly among them. For the hooked numpy functionalities, keep 
in mind that they act on the axis=1 of an internal 2D numpy array
with shape=(1+num_bins, T),
where 'num_bins' is the number of bins,
'1' accounts for the mean value
and 'T' is the lenght of the data (typically the time extent of a correlator).

.. code-block:: python

  from labc import stats
  from labc import data as dt

  num_config = 100
  T = 64
  statsjack = stats.StatsType.Jack(num_config=num_config)

  # 2D array with shape=(num_config, T) containing the raw data
  array_raw_in = np.random.normal(size=(num_config, T))
  #resample
  mean, err, bins = stats.generate_stats(array_raw_in)
  
  # initialize correlator
  corr = dt.DataStats(mean, bins, statsjack)


DataBins
--------

.. autoclass:: labc.data.DataBins
   :members:


DataStats
---------
.. autoclass:: labc.data.DataStats
   :members:
   :inherited-members:


DataErr
-------

.. autoclass:: labc.data.DataErr
   :members:


Utilities
---------

.. autofunction:: labc.data.merge

.. autofunction:: labc.data.zeros

.. autofunction:: labc.data.ones

.. autofunction:: labc.data.empty

.. autofunction:: labc.data.constant

.. autofunction:: labc.data.gaussian

.. autofunction:: labc.data.uniform

.. autofunction:: labc.data.Z2


Decorators
----------

.. autofunction:: labc.data.dataStats_args

.. autofunction:: labc.data.dataStats_vectorized_args

.. autofunction:: labc.data.dataStats_func