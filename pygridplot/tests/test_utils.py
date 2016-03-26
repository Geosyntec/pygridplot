from textwrap import dedent

import nose.tools as nt
import numpy.testing as nptest

import pandas
from six import StringIO

from pygridplot import utils


class test_addSecondColumnLevel(object):
    def setup(self):
        self.testcsv = StringIO(dedent("""\
        Date,A,B,C,D
        X,1,2,3,4
        Y,5,6,7,8
        Z,9,0,1,2
        """))
        self.data = pandas.read_csv(self.testcsv, index_col=['Date'])
        self.known = pandas.MultiIndex.from_tuples([(u'test', u'A'), (u'test', u'B'),
                                                    (u'test', u'C'), (u'test', u'D')])

    def test_normal(self):
        newdata = utils.addSecondColumnLevel('test', 'testlevel', self.data)
        nt.assert_list_equal(self.known.tolist(), newdata.columns.tolist())

    @nptest.raises(ValueError)
    def test_error(self):
        newdata1 = utils.addSecondColumnLevel('test1', 'testlevel1', self.data)
        newdata2 = utils.addSecondColumnLevel('test2', 'testlevel2', newdata1)
