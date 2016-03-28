from __future__ import division
import datetime
import gc

import numpy as np
import pandas
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import ScalarFormatter
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
import shapefile

sns.set(style='ticks', context='paper')

def _rotate_tick_labels(ax):
    '''
        Private function to rotate x-tick labels of an axes

        Input
        -----
        ax : matplotlib.axes.Axes instance

        Output
        ------
        None
    '''
    for label in ax.get_xticklabels():
        label.set_rotation_mode('anchor')
        label.set_rotation(45)
        label.set_horizontalalignment('right')

def _read_out(filename, valcol, nrows=None, velocity=False, icol='I_MOD',
              jcol='J_MOD', tcol='DUMPID', hcol='ST_hr'):
    if velocity:
        names = [tcol, hcol ,'END_hr',icol ,
                 jcol ,'LAYER1' ,'LAYER2' ,'LAYER3' ,
                 'LAYER4' ,'LAYER5' ,'LAYER6' ,'LAYER7' ,
                 'LAYER8' ,'LAYER9' ,'LAYER10                       ']
        output = pandas.read_csv(filename, nrows=nrows, index_col=False,
            skiprows=2, header=None, names=names)
    else:
        output = pandas.read_csv(filename, nrows=nrows, index_col=False, skiprows=1)

    newcols = {icol: 'I', jcol: 'J', tcol: 'tstep', valcol: 'value'}
    output = output.rename(columns=newcols).set_index(['I', 'J', 'tstep'])

    if velocity:
        output.loc[:, ['value']]
    return output

def readModelGrid(shapefilename, icol='MODI', jcol='MODJ', ijcol_idx=[4, 5]):
    '''
        Read in the model grid into a pandas dataframe

        Input
        -----
        shapefilename : string
            filename and path to the shapefile (extension is optional)

        icol : string (default = 'MODI')
            name of the column in the shapefile's attribute table for the "x"
            coordinate of the model grid

        jcol : string (default = 'MODI')
            name of the column in the shapefile's attribute table for the "y"
            coordinate of the model grid

        ijcol_idx : sequence of ints (default = [4, 5])
            positions of `icol` and `jcol` in the attribute table

        Output
        ------
        grid : pandas.dataframe
            index = integer-based multi-index on the values of `icol` and `jcol`
            columns = [
                cell = shapely.geometry.Polygon object reprsenting the grid cell
                area = float value of the area of the grid cell (based on the
                    shapefiles's coordinate system)
    '''
    # read the file
    shpfile = shapefile.Reader(shapefilename)

    # grab field names (first element (0) in everything but the first row (1:))
    fieldnames = np.array(shpfile.fields)[1:, 0]

    # create a dataframe from the records and field names
    grid = pandas.DataFrame(np.array(shpfile.records()), columns=fieldnames)

    # convert I, J to ints
    grid['I'] = grid.apply(lambda row: int(row[icol]), axis=1)
    grid['J'] = grid.apply(lambda row: int(row[jcol]), axis=1)

    # set the index in the model grid locations
    grid.set_index(['I', 'J'], inplace=True)

    # initialize patch column
    grid['cell'] = Polygon([(0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)])

    # loop through all of the shapes and records concurrently
    for shprec in shpfile.shapeRecords():
        shape, row = shprec.shape, shprec.record

        # again, need to pull the MODI/J columns to determine index
        I = row[ijcol_idx[0]]
        J = row[ijcol_idx[1]]

        # set the `cell` column for the row to the actual shape
        grid.loc[I, J]['cell'] = Polygon(shape.points)
    # compute easting and northings from the shapes
    grid['easting'] = grid.apply(lambda row: row['cell'].centroid.x, axis=1)
    grid['northing'] = grid.apply(lambda row: row['cell'].centroid.y, axis=1)

    # return the columns we need
    return grid[['cell', 'area', 'easting', 'northing']]

def attachAnimateValues(grid, filename, valcol, year, month, icol='I_MOD', jcol='J_MOD',
                       tcol='DUMPID', hcol='ST_hr', nrows=None, newfiletype=False,
                       resample_out=None, velocity=False):
    '''
        Reads an output file and add matplotlib patches to grid dataframe for
        plotting

        Input
        -----
        grid : pandas.DataFrame
            output from `readModelGrid(...)`

        filename : string
            full path to the model output file

        icol : string (default = ' icell' # note the leading space)
            name of the column with I grid cell index

        jcol : string (default = ' jcell' # note the leading space)
            name of the column with J grid cell index

        tcol : string (default = ' time_stamp' # note the leading space)
            name of the column containing the timestamp

        valcol : string (default = ' value' # note the leading space)
            name of the column containing the actual values of the output file

        cmap : matplotlib.colors.Colormap instance (default = plt.cm.Blues)
            colormap to be applied to the values when plotting


        Output
        ------
        joined : pandas.dataframe
            index = integer-based multi-index on the values of I and J
            columns = [
                cell = shapely.geometry.Polygon object reprsenting the grid cell
                area = float value of the area of the grid cell (based on the
                    shapefiles's coordinate system)
                value = output value read in from `filename`
                patch = matplotlib.patches.PathPatch to display each cell
                easting = local coordinate easting of the grid cell's centroid
                northing = local coordinate northing of the grid cell's centroid
    '''

    output = _read_out(filename, valcol, nrows=nrows, velocity=velocity, icol=icol,
                  jcol=jcol, tcol=tcol, hcol=hcol)

    data = output[['value', hcol]]
    data.reset_index(inplace=True)
    data.dropna(subset=['tstep'], inplace=True)
    data['datetime'] = (data[hcol].apply(
        lambda dt: datetime.datetime(
            year, month, 1) + datetime.timedelta(dt/24)))
    if resample_out is not None:
        data = (data
                    .set_index('datetime')
                    .groupby(['I', 'J'])
                    .resample(resample_out, how='mean')
                    .reset_index()
                )

    # join the output data and drop NA values (i.e. cells with no output data)
    joined = grid.join(data.set_index(['I', 'J']), how='outer').dropna()
    # joined.set_index(['tstep'], append=True, inplace=True)
    # normalize all of the values
    joined['normed_value'] = (joined.value - joined.value.min()) / joined.value.max()

    # little function to help me make polygon patches for plotting
    def makePatch(row):
        '''
        Input `row` is just a row of a pandas.DataFrame
        '''
        # rgb = (row['r'], row['b'], row['g'])
        patch = PolygonPatch(row['cell'], edgecolor='White', linewidth=0.25)
        return patch

    # add a matplotlib patch to each row
    joined['patch'] = joined.apply(makePatch, axis=1)
    joined = joined.reset_index().set_index(['tstep', 'I', 'J'])
    return joined

def plotGrid(grid, patchcol='patch', ax=None, cmap=plt.cm.Blues, vextent=None,
             log=True, blankgrid=False, **figkwargs):
    '''
        Creates a matplotlib figure of model grid with some output values assoicated
        with each cell

        Input
        -----
        grid : pandas.DataFrame
            ideally, this is output from attachOutputValues(...)

        patchcol : string (default = 'patch')
            name of the column contain the matplotlib.patches.PathPatches in `grid`

        ax : matplotlib.axes.Axes instance for None (default = None)
            optional axes on which to the plot the data. If None, one will be made.

        Output
        ------
        fig : matplotlib.figure.Figure instance
            figure containing the plot
    '''
    figsize = figkwargs.pop('figsize', (6.0, 6.0))
    # check the value of `ax`; create if necessary
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, **figkwargs)
    elif isinstance(ax, plt.Axes):
        fig = ax.figure
    else:
        raise ValueError("`ax` must be None or an MPL Axes object")

    # set the axes aspect ratio and limits based on the data
    # TODO: this needs to be based on extents, not centroids
    ax.set_xlim((grid.easting.min()*.999, grid.easting.max()*1.001))
    ax.set_ylim((grid.northing.min()*.999, grid.northing.max()*1.001))
    ax.set_aspect('equal')

    # create a normalization object based on the data
    if vextent is None:
        norm = plt.Normalize(vmin=grid.value.min(), vmax=grid.value.max())
        if log:
            norm = matplotlib.colors.LogNorm(vmin=grid.value.min()+1E-10, vmax=grid.value.max())
    else:
        norm = plt.Normalize(vmin=np.min(vextent), vmax=np.max(vextent))
        if log:
            norm = matplotlib.colors.LogNorm(vmin=np.min(ve+1E-10), vmax=np.max(ve))
    # create a ScalarMappable based on the normalization object
    # (this is what the colorbar will be based off of)
    cmap.set_under('white')
    cmap.set_bad('white')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # and set it's values to the grid.values.column (may be unnecessary)
    sm._A = np.array(grid.value.tolist())

    # stuff the `patches` (grid cells) column of the grid in a PatchCollection
    edgecol = sm.to_rgba(grid.value.values)*0 + .7

    if not blankgrid:
        facecolors = grid.value.values
    else:
        facecolors = np.zeros(grid.value.values.shape)

    patches = PatchCollection(grid.patch.tolist(), match_original=False,
        facecolors=sm.to_rgba(grid.value.values), linewidths=[.25,.25,.25],
            edgecolors=edgecol)

    # plot the grid cells on the axes
    ax.add_collection(patches)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    # add ad horizontal colorbar (defaults to bottom of figure)
    plt.colorbar(sm, orientation='vertical', cax=cax)

    # format the tick labels
    fmt = ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    _rotate_tick_labels(ax)
    sns.despine()
    time_text = ax.text(0.8, 0.9, '', transform=ax.transAxes, fontsize=8,
            verticalalignment='top')
    textstr = grid['datetime'].iloc[0].strftime("%Y-%m-%d")
    time_text.set_text(textstr)
    # snug up the figure's layout
    fig.tight_layout()

    return fig


class GridAesthetics(object):
    """
    Class to manage shapefile and plotting values.
    """
    def __init__(self, datapath, valcol, shapefile, year, month, icol='I_MOD',
            jcol='J_MOD', tcol='DUMPID', gridicol='EFDC_I', gridjcol='EFDC_J',
            hcol='ST_hr', ijcol_idx=[4, 5], newfiletype=False, resample_out=None,
            velocity=False, u_path=None, v_path=None):

        self.shapefile = shapefile
        self.datapath = datapath
        self._u_path = u_path
        self._v_path = v_path
        self.year = year
        self.month = month
        self.newfiletype = newfiletype
        self.velocity = velocity
        self.resample_out = resample_out

        self._gridicol = gridicol
        self._gridjcol = gridjcol
        self._ijcol_idx = ijcol_idx
        self._valcol = valcol
        self._icol = icol
        self._jcol = jcol
        self._tcol = tcol
        self._hcol = hcol

        self._modelGrid = None
        self._gridValues = None
        self._uv = None

    @property
    def modelGrid(self):
        if self._modelGrid is None:
            gridData = readModelGrid(
                self.shapefile,
                icol=self._gridicol,
                jcol=self._gridjcol,
                ijcol_idx=self._ijcol_idx
            )
            self._modelGrid = gridData
        return self._modelGrid

    @property
    def gridValues(self):
        if self._gridValues is None:
            gv = attachAnimateValues(self.modelGrid, self.datapath,
                self._valcol, self.year, self.month, icol=self._icol, jcol=self._jcol,
                tcol=self._tcol, nrows=None, newfiletype=self.newfiletype,
                resample_out=self.resample_out, velocity=self.velocity
            )
            self._gridValues = gv
        return self._gridValues

    @property
    def uv_values(self):
        if self._uv is None:
            u = _read_out(self._u_path, self._valcol, velocity=True, icol=self._icol,
                          jcol=self._jcol, tcol=self._tcol, hcol=self._hcol)
            v = _read_out(self._v_path, self._valcol, velocity=True, icol=self._icol,
                          jcol=self._jcol, tcol=self._tcol, hcol=self._hcol)
            uv = u.join(v, how='inner', lsuffix='_u', rsuffix='_v')
            uv['value'] = np.sqrt(uv.value_u**2 + uv.value_v**2)

            uv = (uv.reset_index(level='tstep', drop=False)
                    .join(self.modelGrid, how='outer')
                    .dropna()
                    .set_index('tstep', append=True)
            )
            self._uv = uv
        return self._uv

    def plot(self, timestep, ax=None, cmap=plt.cm.Blues, vextent=None,
        log=True, blankgrid=False, **figkwargs):

        fig = plotGrid(self.gridValues.xs(timestep, level='tstep'),
                       ax=ax,cmap=cmap, vextent=vextent, log=log,
                       blankgrid=blankgrid, **figkwargs)
        return fig

    def add_plot_vectors(self, ax, timestep, scale=9e-4,
                  alpha=0.7, headwidth=3.5):
        subset = self.uv_values.xs(timestep, level='tstep')

        ax.quiver(subset.easting, subset.northing,
                  subset.value_u, subset.value_v,
                  angles='xy', scale_units='xy', scale=scale,
                  alpha=alpha, headwidth=headwidth)

        return ax.figure
