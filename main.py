# -*- coding: UTF-8 -*-

if __name__ == '__main__':
    from HistogramEqualization import *
    histogramEqualization = HistogramEqualization('pout.tif')
    histogramEqualization.calculate(True)
