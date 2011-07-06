# Copyright (c) 2011, Chandler Armstrong (omni dot armstrong at gmail dot com)
# see LICENSE.txt for details
# <http://tixxit.net/2010/03/graham-scan/> provides the basis for the code in
# this file.  the code at that address contains no copyright notice.




"""find the convex hull of a set of points using graham scan"""




from numpy import array, in1d, lexsort




TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)




def _turn((p_x, p_y), (q_x, q_y), (r_x, r_y)):
    return cmp((q_x - p_x) * (r_y - p_y) - (r_x - p_x) * (q_y - p_y), 0)


def _keep_left(hull, r):
    while len(hull) > 1 and _turn(hull[-2], hull[-1], r) != TURN_LEFT: hull.pop()
    if not len(hull) or not (hull[-1] == r).all(): hull.append(r)
    return hull


def _sorted(P):
    ind = lexsort((P[:,1],P[:,0]))
    return P[ind]


def main(P):
    """
    Returns an array of the points in convex hull of P in CCW order.

    arguments: P -- a Polygon object or an numpy.array object of points
    """
    P = _sorted(P)
    l = reduce(_keep_left, P, [])
    u = reduce(_keep_left, reversed(P), [])
    l.extend(u[i] for i in xrange(1, len(u) - 1))
    return array(l)
