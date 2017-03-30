'''
Copyright (C) 2014 New York University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np

class NYUDepthModelDefs(object):
    def __init__(self):
        '''
        precomputed means and stdev
        '''

        self.vgg_image_mean = np.array((123.68, 116.779, 103.939), dtype=np.float32)
        self.images_mean = 109.31410628
        self.images_std = 76.18328376
        self.images_istd = 1.0 / self.images_std
        self.depths_mean = 2.53434899
        self.depths_std = 1.22576694
        self.depths_istd = 1.0 / self.depths_std
        self.logdepths_mean = 0.82473954
        self.logdepths_std = 0.45723134
        self.logdepths_istd = 1.0 / self.logdepths_std
        self.semantic_class_num = 40
