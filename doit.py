#!/usr/bin/env python
# coding=utf8

# Copyright (c) 2017 Behrooz Vedadian
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
Sample project demonstrating classification using a straight line
"""

from __future__ import print_function, division, unicode_literals

import codecs
import sys
import cv2
import numpy as np
from matplotlib import pyplot
import cvxpy

# Make the program look alike in python versions 3 and 2
if sys.version_info < (3, 0):
    sys.stdin = codecs.getreader("utf8")(sys.stdin)
    sys.stdout = codecs.getwriter("utf8")(sys.stdout)
    sys.stderr = codecs.getwriter("utf8")(sys.stderr)

def main():
    """ Main function of the script """

    def get_tribe_points(map, channel):
        """Finds center of contours of the desired channel in the map and returns them"""
        _, mass = cv2.threshold(map[:, :, channel], 50, 255, cv2.THRESH_BINARY_INV)
        pyplot.imshow(mass)
        blocks, *_ = cv2.findContours(mass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for block in blocks:
            points.append(np.mean(block.reshape(block.shape[0], -1), 0).reshape(1, -1))
        return np.concatenate(points)

    # Read the original scatter map of tribe homes
    map = cv2.imread('./scatter-map.jpg')

    # Plot the image
    pyplot.subplot(2, 2, 1)
    pyplot.imshow(map)

    # Get the point coordinates from the input image for red tribe
    pyplot.subplot(2, 2, 2)
    red_points = get_tribe_points(map, 0)
    # Get the point coordinates from the input image for blue tribe
    pyplot.subplot(2, 2, 3)
    blue_points = get_tribe_points(map, 2)

    # Set up the optimization problem
    a = cvxpy.Variable()
    b = cvxpy.Variable()
    c = cvxpy.Variable()

    objective = cvxpy.Minimize(a ** 2 + b ** 2)
    constraints = []
    for p in red_points:
        constraints.append(-1 * (a * p[1] + b * p[0] + c + 1) > 0)
    for p in blue_points:
        constraints.append(+1 * (a * p[1] + b * p[0] + c - 1) > 0)
    problem = cvxpy.Problem(objective, constraints)
    problem.solve()

    # Get the optimal values
    a = a.value
    b = b.value
    c = c.value

    # Draw the border and support lines on the original scatter map
    separated_map = np.copy(map)
    def drawline(c, l):
        """Draws a line with given c (f(x) = (b*x+c)/a) with given level of luminance"""

        def evaluate(a, b, c, x):
            """Evaluates the f(x)=(b*x+c)/a"""
            return (x, int(np.round(-(b * x + c) / a)))

        p0 = evaluate(a, b, c, -map.shape[1])
        p1 = evaluate(a, b, c, 2 * map.shape[1])
        cv2.line(separated_map, p0, p1, (l, l, l), 5)

    drawline(c - 1, 100)
    drawline(c + 1, 100)
    drawline(c, 0)

    # Display the map with lines on it
    pyplot.subplot(2, 2, 4)
    pyplot.imshow(separated_map)

    # Show the resulting plot
    pyplot.show()

if __name__ == "__main__":
    main()
