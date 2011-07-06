# Copyright (c) 2011, Chandler Armstrong (omni dot armstrong at gmail dot com)
# see LICENSE.txt for details




"""
polygon object
"""




from __future__ import division
from math import cos, sin, sqrt, radians
from operator import mul

from numpy import array, vstack
from pygame import Rect

from ._convexhull import main as convexhull




# error tolerances
_MACHEPS = pow(2, -24)
_E = _MACHEPS * 10




# look up tables (LUTs)
_cos_table = dict([(deg, cos(radians(deg))) for deg in xrange(0, 360)])
_sin_table = dict([(deg, sin(radians(deg))) for deg in xrange(0, 360)])




# utility functions
_perp = lambda (x, y): array([-y, x])                   # perpendicular
_prod = lambda X: reduce(mul, X)                        # product
_dot = lambda p, q: sum(_prod(X) for X in zip(p, q))    # dot product
_mag = lambda (x, y): sqrt(x * x + y * y)               # magnitude, or length
_normalize = lambda V: array([i / _mag(V) for i in V])  # normalize a vector
_intersect = lambda A, B: (A[1] > B[0] and B[1] > A[0]) # intersection test
_unzip = lambda zipped: zip(*zipped)                    # unzip a list of tuples


def _v(C, q=array([0, 0]), i=0):
    # find the point on the convex hull of C closest to q by iteratively
    #   searching around voronoi regions
    # C is a Polygon object whose points are in CCW order
    # i is the index of the initial test edge
    # returns the point closest to q and the minimum set of points such that q
    #   in conv(points)
    edges, n, P = C.edges, C.n, C.P
    if n == 1: return P[0], set([tuple(P[0])])
    checked, inside = set(), set()
    while 1:
        checked.add(i)
        edge = edges[i]
        p = P[i]
        v = p - q               # vector from p0 to q
        len2 = _dot(edge, edge) # len(edge)**2
        vprj = _dot(v, edge)    # v projected onto edge
        if vprj < 0:    # q lies CW of edge
            i = (i - 1) % n
            if i in checked:
                if not i in inside:
                    return p, set([tuple(p)])
                i = (i - 1) % n
            continue
        if vprj > len2: # q lies CCW of edge
            i = (i + 1) % n
            if i in checked:
                if not i in inside:
                    p = P[i]
                    return p, set([tuple(p)])
                i = (i + 1) % n
            continue
        nprj = _dot(v, _perp(edge)) # v projected onto edge normal
        # perp of CCW edges will always point "outside"
        if nprj > 0: # q is "inside" the edge
            inside.add(i)
            if len(checked) == n: # q is inside C
                return q, set([tuple(p) for p in C])
            i = (i + 1) % n
            continue
        edge_P = set([tuple(p), tuple(P[(i + 1) % n])])
        edge_n = _normalize(edge)
        # move from p to q projected on to edge
        qprj = p - ((_dot(v, edge_n)) * edge_n)
        return qprj, edge_P




class Main(object):
    """polygon object"""


    def __init__(self, P, deg = 0, conv=True):
        """
        arguments:
        P -- iterable or 2d numpy.array of (x, y) points.  the constructor will
          find the convex hull of the points in CCW order; see the conv keyword
          argument for details.

        keyword arguments:
        deg -- degrees of rotation, defaults to 0

        conv -- boolean indicating if the convex hull of P should be found.
          conv is True by default.  Polygon is intended for convex polygons only
          and P must be in CCW order.  conv will ensure that P is both convex
          and in CCW.  even if P is already convex, it is recommended to leave
          conv True, unless client code can be sure that P is also in CCW order.
          CCW order is requried for certain operations.

          NOTE: the order must be with respect to a bottom left orgin; graphics
            applications typically use a topleft origin.  if your points are CCW
            with respect to a topleft origin they will be CW in a bottomleft
            origin
        """
        P = array(list(P))
        if conv: P = convexhull(P)
        self.P = P
        self.deg = deg
        n = len(P) # number of points
        self.n = n
        self.a = self._A() # area of polygon

        edges = [] # an edge is a vector of distance from p to q
        for i, p in enumerate(P):
            q = P[(i + 1) % n] # x, y of next point in series
            edges.append(p - q)
        self.edges = array(edges)


    def __len__(self): return self.n


    def __getitem__(self, i): return self.P[i]

    
    def __iter__(self): return iter(self.P)


    def __repr__(self): return str(self.P)


    def __add__(self, other):
        """
        returns the minkowski sum of self and other

        arguments:
        other is a Polygon object

        returns an array of points for the results of minkowski addition

        NOTE: use the unary negation operator on other to find the so-called
          minkowski difference. eg A + (-B)
        """
        P, Q = self.rotopoints, other.rotopoints
        return array([p + q for p in P for q in Q])


    def __neg__(self): return Main(-self.P, self.deg)


    def get_rect(self):
        """return the AABB, as a pygame rect, of the polygon"""
        X, Y = _unzip(self.P)
        x, y = min(X), min(Y)
        w, h = max(X) - x, max(Y) - y
        return Rect(x, y, w, h)


    def move(self, x, y):
        """return a new polygon moved by x, y"""
        return Main([(x + p_x, y + p_y) for (p_x, p_y) in self.P], self.deg)


    def move_ip(self, x, y):
        """move the polygon by x, y"""
        self.P = array([(x + p_x, y + p_y) for (p_x, p_y) in self.P])


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
        for i, p in enumerate(self.rotopoints):
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

        arguments:
        other -- a polygon object

        returns:
        an array of projections
        """
        # a projection is a vector representing the span of a polygon projected
        # onto an axis
        projections = []
        for edge in vstack((self.rotoedges, other.rotoedges)):
            edge = _normalize(edge)
            # the separating axis is the line perpendicular to the edge
            axis = _perp(edge)
            self_projection = self.project(axis)
            other_projection = other.project(axis)
            # if self and other do not intersect on any axis, they do not
            # intersect in space
            if not _intersect(self_projection, other_projection): return False
            # find the overlapping portion of the projections
            projection = self_projection[1] - other_projection[0]
            projections.append((axis[0] * projection, axis[1] * projection))
        return array(projections)


    def _s(self, C):
        # returns a function that returns the support mapping of C
        # the support mapping is the p in C such that
        #   dot(r, p) == dot(r, _s(C)(r))
        # ie, the support mapping is the p in C most in the direction of r
        return lambda r: max(dict((_dot(r, p), p) for p in C).items())[1]


    def _support(self, P, Q):
        # returns a function that returns the support mapping of P - Q; s_P-Q
        # s_P-Q is the generic support mapping for polygons
        # NOTE: return type of the returned function is a tuple
        s = self._s
        s_P, s_Q = s(P), s(Q)
        return lambda r: tuple(s_P(r) - s_Q(-r))


    def distance(self, other, r=array([0, 0])):
        """
        return distance between self and other
        uses GJK algorithm. for details see:

        Bergen, Gino Van Den. (1999). A fast and robust GJK implementation for
          collision detection of convex objects. Journal of Graphics Tools 4(2).

        arguments:
        other -- a Polygon object

        keyword arguments
        r -- initial search direction; setting r to the movement vector of
          self - other may speed convergence
        """
        P, Q = self.rotopoints, other.rotopoints
        support = self._support(P, Q) # support mapping function s_P-Q(r)
        v = array(support(r)) # support point
        W = set()
        w = support(-v)
        while _dot(v, v) - _dot(w, v) > _MACHEPS: # while w is closer to origin
            Y = Main(W.union([w]))
            v, W = _v(Y) # point and smallest W containing closest point in Y
            if len(W) == 3: return v # the origin is inside W; intersection
            w = support(-v)
        return v


    def raycast(self, other, r, s=array([0, 0])):
        """
        return the hit scalar, hit vector, and hit normal from self to other in
          direction r
        uses GJK-based raycast.  for details see:

        Bergen, Gino Van Den. (2004). Ray casting against general convex
          objects with application to continuous collision detection. GDC 2005.
          retrieved from
          http://www.bulletphysics.com/ftp/pub/test/physics/papers/
            jgt04raycast.pdf
          on 6 July 2011.

        arguments:
        other -- Polygon object
        r -- direction vector
          NOTE: GJK searches IN THE DIRECTION of r, thus r needs to point
            towards the origin with respect to the direction vector of self; in
            other words, if r represents the movement of self then client code
            should call raycast with -r.

        keyword arguments:
        s -- initial position along r, (0, 0) by default

        returns:
        if r does not intersect other, returns False
        else, returns the hit scalar, hit vector, and hit normal
          hit scalar -- the scalar where r intersects other
          hit vector -- the vector where self intersects other
          hit normal -- the edge normal at the intersection
        """
        P, Q = self.rotopoints, other.rotopoints

        support = self._support(P, Q) # support mapping function s_P-Q(r)

        lambda_ = 0               # scalar of r to hit spot
        q = s                     # current point along r
        n = 0                     # the normal at q
        v = q - array(support(r)) # vector from q to s_P-Q
        P = set()

        p = support(v)
        P.add(p)
        w = q - p
        while _dot(v, v) > _E * max(_dot(q - p, q - p) for p in P):
            if _dot(v, w) > 0:
                if _dot(v, r) >= 0: return False
                lambda_ = lambda_ - (_dot(v, w) / _dot(v, r))
                q = s + (lambda_ * r)
                n = v
            if lambda_ > 1: return False
            Y = Main(P)
            v, P = _v(Main(q - Y))   # _v is working with P in negative space
            P = set([(-x, -y) for x, y in P]) # switch P back to positive space

            p = support(v)
            P.add(p)
            w = q - p
        return lambda_, q, n


    def _A(self):
        # the area of polygon
        n = self.n
        X, Y = _unzip(self.P)
        return 0.5 * sum(X[i] * Y[(i + 1) % n] - X[(i + 1) % n] * Y[i]
                         for i in xrange(n))


    @property
    def C(self):
        """returns the centroid of the polygon"""
        a, n = self.a, self.n
        X, Y = _unzip(self.P)
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
        # returns coords of point (x, y) rotated self.deg degrees around the
        # origin
        deg = self.deg
        if origin: o_x, o_y = origin
        else: o_x, o_y = self.C
        costheta = _cos_table[deg]
        sintheta = _sin_table[deg]
        return ((o_x + costheta * (x - o_x)) - (sintheta * (y - o_y)),
                (o_y + sintheta * (x - o_x)) + (costheta * (y - o_y)))


    @property
    def rotopoints(self):
        """
        returns an array of points rotated self.deg degrees around the
        centroid
        """
        P = self.P
        rotate = self._rotate
        return array([rotate(p) for p in P])


    @property
    def rotoedges(self):
        """return an array of vectors of edges rotated self.deg degrees"""
        edges = self.edges
        rotate = self._rotate
        # edges, essentially angles, are always rotated around the origin (0, 0)
        return array([rotate(edge, (0, 0)) for edge in edges])


    def project(self, axis):
        """project self onto axis"""
        P = self.rotopoints
        projected_points = [_dot(p, axis) for p in P]
        # return the span of the projection
        return min(projected_points), max(projected_points)
