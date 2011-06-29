# Copyright (c) 2011, Chandler Armstrong (omni dot armstrong at gmail dot com)
# see LICENSE.txt for details



"""
polygon object
"""



from __future__ import division
from math import cos, sin, sqrt, radians
from operator import mul

from pygame import Rect



prod = lambda X: reduce(mul, X) # product
dot = lambda X, Y: prod(X) + prod(Y) # dot product
mag = lambda (x, y): sqrt(x * x + y * y) # the magnitude, or length
normalize = lambda mag, V: [i / mag for i in V] # normalize a vector
intersect = lambda A, B: (A[1] > B[0] and B[1] > A[0]) # intersection test
unzip = lambda zipped: zip(*zipped) # unzip a list of tuples


# look up tables (LUTs)
cos_table = dict([(deg, cos(radians(deg))) for deg in xrange(0, 360)])
sin_table = dict([(deg, sin(radians(deg))) for deg in xrange(0, 360)])



class Polygon(object):
    """polygon object"""


    def __init__(self, P, deg = 0):
        """
        P -- tuple of (x, y) points representing vertices and edges of polygon.
             edges are created via the order of points in the tuple; eg. point
             0 and point 1 form an edge, point 1 and point 2...  an edge is
             automatically created between the last point and point 0; do not
             include the vertex at point 0 as also the last point in the tuple.
             the polygon must be convex; edges cannot cross or form concavities.
        deg -- degrees of rotation, defaults to 0
        """
        self.deg = deg
        self.P = P
        n = len(P) # number of points
        self.n = n
        self.a = self._A() # area of polygon

        edges = [] # an edge is a vector of distance from p to q
        for i, p in enumerate(P):
            p_x, p_y = p
            q_x, q_y = P[(i + 1) % n] # x, y of next point in series
            edges.append((q_x - p_x, q_y - p_y))
        self.edges = edges


    def __len__(self): return self.n


    def __getitem__(self, i): return self.P[i]

    
    def __iter__(self): return iter(self.P)


    def __repr__(self): return str(self.P)


    def get_rect(self):
        """return the AABB, as a pygame rect, of the polygon"""
        X, Y = unzip(self.P)
        x, y = min(X), min(Y)
        w, h = max(X) - x, max(Y) - y
        return Rect(x, y, w, h)


    def move(self, x, y):
        """return a new polygon moved by x, y"""
        return Polygon([(x + p_x, y + p_y) for (p_x, p_y) in self.P], self.deg)


    def move_ip(self, x, y):
        """move the polygon by x, y"""
        self.P = [(x + p_x, y + p_y) for (p_x, p_y) in self.P]


    def collidepoint(self, (x, y)):
        """
        test if point (x, y) is outside, on the boundary, or inside polygon
        uses raytracing algorithm

        returns 0 if outside
        returns -1 if on boundary
        returns 1 if inside
        """
        n, P = self.n, self.P

        # test if (x, y) on a vertex
        for p_x, p_y in P:
            if (x == p_x) and (y == p_y): return -1

        intersections = 0
        for i, p in enumerate(P):
            p_x, p_y = p
            q_x, q_y = P[(i + 1) % n]
            x_min, x_max = min(p_x, q_x), max(p_x, q_x)
            y_min, y_max = min(p_y, q_y), max(p_y, q_y)
            # test if (x, y) on horizontal boundary
            if (p_y == q_y) and (p_y == y) and (x > x_min) and (x < x_max):
                return -1
            if (y > y_min) and (y <= y_max) and (x <= x_max) and (p_y != q_y):
                x_inters = (((y - p_y) * (q_x - p_x)) / (q_y - p_y)) + p_x
                # test if (x, y) on non-horizontal polygon boundary
                if x_inters == x: return -1
                # test if line from (x, y) intersects boundary
                if p_x == q_x or x <= x_inters: intersections += 1

        return intersections % 2


    def collidepoly(self, other):
        """
        test if other polygon collides with self using seperating axis theorem
        if collision, return projections
        """
        # get all edges, from both polygons
        edges = [normalize(mag(edge), edge) for edge in (self.rotoedges + other.rotoedges)]
        # a projection is a vector representing the span of a polygon projected onto an axis
        projections = []
        for edge in edges:
            x, y = -edge[1], edge[0] # the separating axis is the line perpendicular to the edge
            self_projection = self.project((x, y))
            other_projection = other.project((x, y))
            # if self and other do not intersect on an axis, they do not intersect
            if not intersect(self_projection, other_projection): return False
            # find the overlapping portion of the projections
            projection = self_projection[1] - other_projection[0]
            projections.append((x * projection, y * projection))
        return projections


    def _A(self):
        # the area of polygon
        n = self.n
        X, Y = unzip(self.P)
        return 0.5 * sum(X[i] * Y[(i + 1) % n] - X[(i + 1) % n] * Y[i] for i in xrange(n))


    @property
    def C(self):
        """returns the centroid of the polygon"""
        a, n = self.a, self.n
        X, Y = unzip(self.P)
        c_x, c_y = 0, 0
        for i in xrange(n):
            a_i = X[i] * Y[(i + 1) % n] - X[(i + 1) % n] * Y[i]
            c_x += (X[i] + X[(i + 1) % n]) * a_i
            c_y += (Y[i] + Y[(i + 1) % n]) * a_i
        b = 1 / (6 * a)
        c_x *= b
        c_y *= b
        return c_x, c_y


    @C.setter
    def C(self, (x, y)):
        c_x, c_y = self.C
        x, y = x - c_x, y - c_y
        self.P = [(p_x + x, p_y + y) for (p_x, p_y) in self.P]


    def _rotate(self, (x, y), origin = None):
        # returns coords of point (x, y) rotated self.deg degrees around the origin
        deg = self.deg
        if origin: o_x, o_y = origin
        else: o_x, o_y = self.C
        costheta = cos_table[deg]
        sintheta = sin_table[deg]
        return ((o_x + costheta * (x - o_x)) - (sintheta * (y - o_y)),
                (o_y + sintheta * (x - o_x)) + (costheta * (y - o_y)))


    @property
    def rotopoints(self):
        """returns position of points rotated self.deg degrees around the centroid"""
        P = self.P
        rotate = self._rotate
        return [rotate(p) for p in P]


    @property
    def rotoedges(self):
        """return vectors of edges rotated self.deg degrees"""
        edges = self.edges
        rotate = self._rotate
        # edges, essentially angles, are always rotated around the origin (0, 0)
        return [rotate(edge, (0, 0)) for edge in edges]                       


    def project(self, axis):
        """project self onto axis"""
        P = self.rotopoints
        projected_points = [dot(*zip(p, axis)) for p in P]
        return min(projected_points), max(projected_points) # return the span of the projection



class Line(object):
    """line segment object"""


    def __init__(self, *a):
        """
        construct a line segment object
        arguments must be one of the following:
        two two-place tuples of two numbers each: ((p_x, p_y), (q_x, q_y))
        four numbers: (p_x, p_y, q_x, q_y)
        a Line object
        """
        if len(a) == 1: a = a[0] # a is a Line object
        elif len(a) == 4: a = [(a[0], a[1]), (a[2], a[3])] # (p_x, p_y, q_x, q_y)

        assert len(a) == 2
        for s in a: assert len(s) == 2
        self.p, self.q = a


    def __len__(self): return 2


    def __getitem__(self, i):        
        return (self.p, self.q)[i]

    
    def __iter__(self):        
        return iter(self.p, self.q)


    def __repr__(self):
        return str(self.p) + ', ' + str(self.q)


    @property
    def rect(self):
        """return the AABB, as a pygame rect, of self"""
        p, q = self.p, self.q
        x, y = [min(e) for e in zip(p, q)]
        w = abs(p[0] - q[0])
        h = abs(p[1] - q[1])
        return Rect(x, y, w, h)


    @property
    def m(self):
        """return the slope of self"""
        p_x, p_y = self.p
        q_x, q_y = self.q
        if p_x != q_x:
            return (q_y - p_y) / (q_x - p_x)
        else:
            return None


    @property
    def b(self):
        """return the y-intercept of self"""
        p_x, p_y = self.p
        m = self.m
        if m != None: return p_y - (m * p_x)
        else: return None


    @property
    def dist(self):
        """return the distance covered, or length, of self"""
        p_x, p_y = self.p
        q_x, q_y = self.q
        return sqrt((q_x - p_x)**2 + (q_y - p_y)**2)


    @property
    def delta(self):
        """return a vector representing the difference between the endpoints of self"""
        p_x, p_y = self.p
        q_x, q_y = self.q
        return (q_x - p_x, q_y - p_y)


    def intersection(self, *a):
        """
        return the point of intersection between self and other
        a must be an argument list that can be used to construct a Line object
        """
        other = Line(*a)
        self_rect, other_rect = self.rect, other.rect
        # test if AABBs intersect
        if not self_rect.colliderect(other_rect): return None
        self_px, other_px = self.p[0], other.p[0]
        self_m, other_m = self.m, other.m
        self_b, other_b = self.b, other.b
        # test that lines are not parallel
        if self_m != other_m:
            if self_m and other_m: # neither line is verticle
                x = (other_b - self_b) / (self_m - other_m)
                y = (self_m * x) + self_b
            elif self_m == None:   # self is verticle, use other
                x = self_px
                y = (other_m * self_px) + other_b
            elif other_m == None:  # other is verticle, use self
                x = other_px
                y = (self_m * other_px) + self_b
            return (x, y)
        elif self_b == other_b: return self # lines are equal
        else: return None # lines are parallel


    def line_clip(self, rect):
        """
        clip self to rect using liang-barsky
        returns a new line object of self clipped to rect
        """
        p_x, p_y = self.p
        delta_x, delta_y = self.delta
        t0, t1 = 0, 1 # initialize min, max scalar

        if delta_x != 0:
            rl = (rect.left - p_x) / delta_x    # clipping scalar to left edge
            rr = (rect.right - p_x) / delta_x   # clipping scalar to right edge
            if delta_x > 0:                     # if p is leftmost point
                if (rl > t0) and (0 <= rl <= 1): t0 = rl
                if (rr < t1) and (0 <= rr <= 1): t1 = rr
            else:                               # else p is rightmost point
                if (rl < t1) and (0 <= rl <= 1): t1 = rl
                if (rr > t0) and (0 <= rr <= 1): t0 = rr

        if delta_y != 0:
            rb = (rect.bottom - p_y) / delta_y  # clipping scalar to bottom edge
            rt = (rect.top - p_y) / delta_y     # clipping scalar to top edge
            if delta_y > 0:                     # if p is topmost point
                if (rb < t1) and (0 <= rb <= 1): t1 = rb
                if (rt > t0) and (0 <= rt <= 1): t0 = rt
            else:                               # else p is bottommost point
                if (rb > t0) and (0 <= rb <= 1): t0 = rb
                if (rt < t1) and (0 <= rt <= 1): t1 = rt

        if t0 > t1: return False

        return Line(p_x + (t1 * delta_x),
                    p_y + (t1 * delta_y),
                    p_x + (t0 * delta_x),
                    p_y + (t0 * delta_y))


    def line_trace(self):
        """return integer coordinates, or pixels, crossed by self"""
        s = set() # coordinates crossed by self
        delta_x, delta_y = self.delta
        p_x, p_y = self.p
        q_x, q_y = self.q
        m, b = self.m, self.b
        s.add((p_x, p_y))

        if (m) or (abs(m) >= 1): # rise > run
            m = 1 / m    # rotate m by 90 degrees
            b = -(b * m) # rotate b by 90 degrees
            # determine direction of p_y to q_y
            if delta_y < 0: delta_y = -1
            else: delta_y = 1
            # trace from p_y to q_y
            while p_y != q_y:
                p_y += delta_y
                s.add((round((m * p_y) + b), p_y))

        else: # else rise < run
            # determine direction of p_x to q_x
            if delta_x < 0: delta_x = -1
            else: delta_x = 1
            # trace from p_x to q_x
            while p_x != q_x:
                p_x += delta_x
                s.add((p_x, round((m * p_x) + b)))

        return s
