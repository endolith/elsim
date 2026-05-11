Benchmarks
==========

Performance of the public API is tracked with `Airspeed Velocity (ASV)`_.

.. _Airspeed Velocity (ASV): https://asv.readthedocs.io/en/stable/

An `interactive ASV report`_ is published alongside this documentation when the default branch is deployed (same GitHub Pages site).

.. _interactive ASV report: ../benchmarks/

The CI job runs on Ubuntu with Python 3.12 and installs optional-dependencies ``fast`` so Numba-backed code paths are exercised. Results accrue on that runner; for local comparisons or additional commits, install optional-dependencies ``bench``, ``cd`` into ``benchmarks/``, and follow the `ASV workflow`_ (configuration is in ``asv.conf.json``).

.. _ASV workflow: https://asv.readthedocs.io/en/stable/using.html
