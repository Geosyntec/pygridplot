# pygridplot

[![Build Status](https://travis-ci.org/Geosyntec/pygridplot.svg?branch=master)](https://travis-ci.org/Geosyntec/pygridplot)
[![Coverage Status](https://coveralls.io/repos/github/Geosyntec/pygridplot/badge.svg?branch=master)](https://coveralls.io/github/Geosyntec/pygridplot?branch=master)

Visualize data over a model grid from an ESRI shapefile.

## Installation
Recommended installation is via conda
```
conda create -n animate python=2.7 pandas seaborn numexpr jupyter
activate animate
pip install descartes
conda install --yes --channel=ioos shapely
conda install --yes --channel=pelson pyshp
```

Then, with the environment still activated run the setup.py or install via pip.
In other words, when you're in the directory with setup.py:

`python setup.py install`

or

`pip install .`
