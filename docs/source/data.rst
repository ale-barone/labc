Data
----

Data are handled by object of the submodules **data**. Once a ``StatsType`` object is initialized,
we can store the resampled bins in a ``DataStats`` objects and perform all the mathematical operation
directly among them. The class also hooks numpy arrays: standard numpy functions can be
directly used on ``DataStats`` objects, keeping in mind that they act on the
axis=1 of an internal 2D numpy array
with shape=(1+num_bins, T),
where 'num_bins' is the number of bins,
'1' accounts for the mean value
and 'T' is the lenght of the data (typically the time extent of a correlator).

.. code-block:: python

  from labc import stats
  from labc import data as dt


  statsjack = stats.StatsType.Jack(num_config=num_config)

  mean, err, bins = stats.generate_stats(array_raw_in)
  # array_row_in is a 2D array with shape shape=(num_config, T) containing the raw data

  corr = dt.DataStats(mean, bins, stats)

.. autoclass:: labc.data.DataStats
  :members:

