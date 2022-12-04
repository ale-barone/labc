Stats
-----

A ``StatsType`` object can be created through the the **stats** submodule.
It can initialize either a bootstrap or a jackknife resampling.

.. code-block:: python

   from labc import stats

   statjack = stats.StatsType.Jack(num_config=num_config)
   # OR
   statsboot = stats.StatsType.Boot(num_config=num_config, num_bins=num_bins, seed=0)


They both inherit from the same private class ``StatsBase`` and share the same methods.
They differ essential for the way the data are resampled (i.e. how the bins are created).

.. autoclass:: labc.stats._statsbase.StatsBase
  :members:

