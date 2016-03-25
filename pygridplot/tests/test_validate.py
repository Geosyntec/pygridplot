import numpy.testing as nptest
import nose.tools as nt
from matplotlib.testing.decorators import image_comparison, cleanup

from matplotlib import pyplot
import pandas

from pygridplot import validate

from six import StringIO


@nt.nottest
def assert_fig_and_ax(fig, ax):
    nt.assert_true(isinstance(fig, pyplot.Figure))
    nt.assert_true(isinstance(ax, pyplot.Axes))


class test__check_ax(object):
    def setup(self):
        fig, ax = pyplot.subplots()
        self.fig = fig
        self.ax = ax

    def teardown(self):
        pyplot.close('all')

    def test_no_ax(self):
        fig, ax = validate.figure_axes(None)
        assert_fig_and_ax(fig, ax)

    def test_ax(self):
        fig, ax = validate.figure_axes(self.ax)
        assert_fig_and_ax(fig, ax)

        nt.assert_equal(fig, self.fig)
        nt.assert_equal(ax, self.ax)

    @nt.raises(ValueError)
    def test_ax_bad_value(self):
        fig, ax = validate.figure_axes('junk')
