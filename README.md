# pygridplot
Visualize data over a model grid from an ESRI shapefile.

## Installation
conda create -n animate python=2.7 pandas seaborn numexpr jupyter
activate animate
pip install descartes
conda install --yes --channel=ioos shapely
conda install --yes --channel=pelson pyshp
