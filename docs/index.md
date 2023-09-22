
# starspot

*starspot* is a tool for measuring stellar rotation periods using
Lomb-Scargle (LS) periodograms, autocorrelation functions (ACFs), phase
dispersion minimization (PDM) and Gaussian processes (GPs).


*starspot* is compatible with any light curve with time, flux and flux
uncertainty measurements, including Kepler, K2 and TESS light curves.
If your light curve is has evenly-spaced (or close to evenly-spaced)
observations, all three of these methods: LS periodograms, ACFs and GPs will
be applicable.
For unevenly spaced light curves like those from Gaia, or ground-based
observatories, LS periodograms and GPs are preferable to ACFs.

For example usage, see the [tutorial notebook](notebooks/tutorial.ipynb).

## License & attribution

Copyright 2018, Ruth Angus. Updates to Jax by Benjamin Pope.

*starspot* uses the [astropy](http://www.astropy.org/) implementation of
[Lomb-Scargle periodograms](http://docs.astropy.org/en/stable/stats/lombscargle.html) and the
[tinygp](https://tinygp.readthedocs.io/) implementation of
fast [celerite](https://celerite.readthedocs.io/en/latest/?badge=latest)
Gaussian processes.

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite this package, its dependencies, and the associated [paper](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.2094A) 

```bibtex
@ARTICLE{Angus2018,
       author = {{Angus}, Ruth and {Morton}, Timothy and {Aigrain}, Suzanne and {Foreman-Mackey}, Daniel and {Rajpaul}, Vinesh},
        title = "{Inferring probabilistic stellar rotation periods using Gaussian processes}",
      journal = {\mnras},
     keywords = {methods: data analysis, methods: statistical, techniques: photometric, stars: rotation, stars: solar-type, starspots, Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2018,
        month = feb,
       volume = {474},
       number = {2},
        pages = {2094-2108},
          doi = {10.1093/mnras/stx2109},
archivePrefix = {arXiv},
       eprint = {1706.05459},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.2094A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```