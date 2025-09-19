Stats
=====

.. automodule:: labc.stats

StatsBase
---------

.. class:: labc.stats.StatsBase

``StatsBase`` is the shared abstract base for all resampling objects.
:class:`StatsType`, :class:`StatsJack`, and :class:`StatsBoot` all inherit
from it. In normal usage you never instantiate ``StatsBase`` directly —
use :meth:`StatsType.Jack` or :meth:`StatsType.Boot` instead.

StatsType
---------

.. autoclass:: labc.stats.StatsType
   :show-inheritance:
   :members:

Jackknife
---------

.. autoclass:: labc.stats.StatsJack
   :show-inheritance:
   :members:
   :inherited-members:


Bootstrap
---------

.. autoclass:: labc.stats.StatsBoot
   :show-inheritance:
   :members:
   :inherited-members:



