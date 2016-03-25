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
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
import seaborn as sns

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

    # return the columns we need
    return grid[['cell', 'area']]

def attachAnimateValues(grid, filename, valcol, year, month, icol='I_MOD', jcol='J_MOD',
                       tcol='DUMPID', timecol='ST_hr', nrows=None, newfiletype=False,
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
    # read the data
    if newfiletype:
        if velocity:
            names = [tcol, timecol ,'END_hr',icol ,
                     jcol ,'LAYER1' ,'LAYER2' ,'LAYER3' ,
                     'LAYER4' ,'LAYER5' ,'LAYER6' ,'LAYER7' ,
                     'LAYER8' ,'LAYER9' ,'LAYER10' ]
            output = pandas.read_csv(filename, nrows=nrows, index_col=False,
                skiprows=2, header=None, names=names)
        else:
            output = pandas.read_csv(filename, nrows=nrows, index_col=False, skiprows=1)
    else:
        output = pandas.read_csv(filename, nrows=nrows, index_col=False)
    newcols = {icol: 'I', jcol: 'J', tcol: 'tstep', valcol: 'value'}
    output = output.rename(columns=newcols).set_index(['I', 'J', 'tstep'])

    data = output[['value', timecol]]
    data.reset_index(inplace=True)
    data.dropna(subset=['tstep'], inplace=True)
    data['datetime'] = (data[timecol].apply(
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

    # compute easting and northings from the shapes
    joined['easting'] = joined.apply(lambda row: row['cell'].centroid.x, axis=1)
    joined['northing'] = joined.apply(lambda row: row['cell'].centroid.y, axis=1)
    joined = joined.reset_index().set_index(['tstep', 'I', 'J'])
    return joined

def animateGrid(grid, frames, patchcol='patch', valuecol='value', ax=None,
    filename=None, interval=250, cmap=plt.cm.Blues, axeslims=None, log=True, **figkwargs):
    '''
        Creates a matplotlib figure of model grid with some output values
        associated with each cell

        Input
        -----
        grid : pandas.DataFrame
            ideally, this is output from attachOutputValues(...)

        patchcol : string (default = 'patch')
            name of the column contain the matplotlib.patches.PathPatches in
            `grid`

        ax : matplotlib.axes.Axes instance for None (default = None)
            optional axes on which to the plot the data. If None, one will be
            made.

        filename : string or None (default = None)
            path to an image where the figure will be saved


        Output
        ------
        fig : matplotlib.figure.Figure instance
            figure containing the plot
    '''
    # check the value of `ax`; create if necessary
    figsize = figkwargs.pop('figsize', (6.0, 6.0))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=300, **figkwargs)
    elif isinstance(ax, plt.Axes):
        fig = ax.figure
    else:
        raise ValueError("`ax` must be None or an MPL Axes object")
    lw=.5 #line width
    # set the axes aspect ratio and limits based on the data
    # TODO: this needs to be based on extents, not centroids

    # setup the axes limits based on the data for the plotter
    if axeslims is None:
        ax.set_xlim((grid.easting.min()*.999, grid.easting.max()*1.001))
        ax.set_ylim((grid.northing.min()*.999, grid.northing.max()*1.001))
    elif axeslims is not None:
        ax.set_xlim(axeslims[0])
        ax.set_ylim(axeslims[1])
    ax.set_aspect('equal')
    # create a normalization object based on the data
    if log:
        norm = matplotlib.colors.LogNorm(vmin=grid[valuecol].min()+1E-10,
            vmax=grid[valuecol].max())
    else:
        norm = plt.Normalize(vmin=grid[valuecol].min(), vmax=grid[valuecol].max())
    # create a ScalarMappable based on the normalization object
    # (this is what the colorbar will be based off of)
     # need to plot a single frame to setup the ploter
    tsteps = grid.index.get_level_values('tstep').unique()
    gridsubset = grid.xs(tsteps[0], level='tstep')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # and set it's values to the grid.values.column (this is necessary)
    sm._A = np.array(gridsubset[valuecol].tolist())

    patches = PatchCollection(gridsubset[patchcol].tolist(),
        match_original=False,
        facecolors=sm.to_rgba(gridsubset[valuecol].values),
        linewidths=[lw,lw,lw], edgecolor='White'
    )
    ax.add_collection(patches)
    # format the tick labels and other plot aesthetics
    fmt = ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    _rotate_tick_labels(ax)
    fig.colorbar(sm, orientation='horizontal')
    sns.despine()
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=8,
            verticalalignment='top')

    def animate(i):
        gridsubset = grid.xs(tsteps[i], level='tstep')
        textstr = gridsubset['datetime'].iloc[0].strftime("%d/%m/%y")
        time_text.set_text(textstr)
        sm._A = np.array(gridsubset[valuecol].tolist())
        patches = PatchCollection(gridsubset[patchcol].tolist(),
            match_original=False,
            facecolors=sm.to_rgba(gridsubset[valuecol].values),
            linewidths=[lw,lw,lw],
            edgecolor='0.1'
        )
        ax.add_collection(patches)
        gc.collect()

    ani = animation.FuncAnimation(fig, animate, frames,
        repeat=False, interval=interval)

    if filename is not None:
        ani.save(filename, fps=30, dpi=300, extra_args=['-vcodec', 'libx264'])
    return ani

def attachPlotValues(grid, filename, timestep, icol=' icell', jcol=' jcell',
                       tcol=' time_step', valcol=' value',
                       cmap=plt.cm.Blues):
    '''
        Reads an output file and add matplotlib patches to grid dataframe for
        plotting

        Input
        -----
        grid : pandas.DataFrame
            output from `readModelGrid(...)`

        filename : string
            full path to the model output file

        timestep : int
            timestep number (must be present in the output file)

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
    # read the data
    output = pandas.read_csv(filename)
    newcols = {icol: 'I', jcol: 'J', tcol: 'tstep', valcol: 'value'}
    output = output.rename(columns=newcols).set_index(['I', 'J', 'tstep'])

    # pull out value column at the timestep (force to DataFrame instead of Series)
    data = pandas.DataFrame(output.xs(timestep, level='tstep')['value'])

    # join the output data and drop NA values (i.e. cells with no output data)
    joined = grid.join(data).dropna()

    # normalize all of the values
    joined['normed_value'] = ((joined.value - joined.value.min()) /
        joined.value.max())

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

    # compute easting and northings from the shapes
    joined['easting'] = joined.apply(lambda row: row['cell'].centroid.x, axis=1)
    joined['northing'] = joined.apply(lambda row: row['cell'].centroid.y, axis=1)

    return joined

def plotGrid(grid, patchcol='patch', ax=None, figname=None, figclose=True,
    cmap=plt.cm.Blues, **figkwargs):
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

        figname : string or None (default = None)
            path to an image where the figure will be saved

        figclose : bool (default = True)
            if True and figname is not None, this will close the figure after being
            saved to disk.

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

    # set the axes aspect ration and limits based on the data
    # TODO: this needs to be based on extents, not centroids
    ax.set_xlim((grid.easting.min()*.999, grid.easting.max()*1.001))
    ax.set_ylim((grid.northing.min()*.999, grid.northing.max()*1.001))
    ax.set_aspect('equal')

    # create a normalization object based on the data
    norm = plt.Normalize(vmin=grid.value.min(), vmax=grid.value.max())

    # create a ScalarMappable based on the normalization object
    # (this is what the colorbar will be based off of)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # and set it's values to the grid.values.column (may be unnecessary)
    sm._A = np.array(grid.value.tolist())

    # stuff the `patches` (grid cells) column of the grid in a PatchCollection
    patches = PatchCollection(grid.patch.tolist(), match_original=False,
        facecolors=sm.to_rgba(grid.value.values), linewidths=[0,0,0])

    # plot the grid cells on the axes
    ax.add_collection(patches)

    # add ad horizontal colorbar (defaults to bottom of figure)
    plt.colorbar(sm, orientation='horizontal')

    # format the tick labels
    fmt = ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    _rotate_tick_labels(ax)
    sns.despine()
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=8,
            verticalalignment='top')
    textstr = grid['datetime'].iloc[0].strftime("%d/%m/%y")
    time_text.set_text(textstr)
    # snug up the figure's layout
    fig.tight_layout()

    # save the figure if possible
    if figname is not None:
        fig.savefig(figname, bbox_inches='tight', dpi=400)

        # close the figure if possible
        if figclose:
            plt.close(fig)

    return fig


class GridAesthetics(object):
    def __init__(self, datapath, valcol, shapefile, year, month, icol='I_MOD',
            jcol='J_MOD', tcol='DUMPID', gridicol='EFDC_I', gridjcol='EFDC_J',
            ijcol_idx=[4, 5], newfiletype=False, resample_out=None, velocity=False):
        self.shapefile = shapefile
        self.datapath = datapath
        self.year = year
        self.month = month
        self.newfiletype = newfiletype
        self.velocity = velocity
        self.resample_out = resample_out

        self._modelGrid = None
        self._gridValues = None
        self._gridicol=gridicol
        self._gridjcol=gridjcol
        self._ijcol_idx=ijcol_idx
        self._valcol = valcol
        self._icol=icol
        self._jcol=jcol
        self._tcol=tcol


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

    def plot(self, timestep, ax=None, figname=None, cmap=plt.cm.Blues):
        fig = plotGrid(self.gridValues.xs(timestep, level='tstep'), ax=ax,
            figname=figname, cmap=cmap)
        return fig

    def animate(self, frames=None, ax=None, filename=None, interval=250,
        cmap=plt.cm.Blues, **figkwargs):
        if frames is None:
            frames = (self.gridValues.index
                .get_level_values('tstep').unique().shape[0])
        anim = animateGrid(self.gridValues, frames=frames, ax=ax,
            filename=filename, interval=interval, cmap=cmap, **figkwargs)
        return anim
