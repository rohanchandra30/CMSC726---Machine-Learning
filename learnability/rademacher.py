from random import randint, seed
from collections import defaultdict
from math import atan, sin, cos, pi, ceil
import matplotlib.pyplot as plt
import tkinter
from numpy import array, single, spacing, divide, correlate, dot, sign, sort, unique, empty, ones, append, mean, isinf, sqrt, prod
from numpy.linalg import norm
from itertools import combinations
from bst import BST
import time
from scipy.spatial import ConvexHull
import functools

start_time = time.time()


kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]
# kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.)]
# kSIMPLE_DATA = []
# for ii in range(3):
#     x = randint(-500,500)
#     y = randint(-500,500)
#     kSIMPLE_DATA.append((x,y))

# print(kSIMPLE_DATA)

class Classifier:
    def correlation(self, data, labels):
        """
        Return the correlation between a label assignment and the predictions of
        the classifier
        Args:
          data: A list of datapoints
          labels: The list of labels we correlate against (+1 / -1)
        """

        assert len(data) == len(labels), \
            "Data and labels must be the same size %i vs %i" % \
            (len(data), len(labels))

        assert all(x == 1 or x == -1 for x in labels), "Labels must be binary"

        # Cast labels as an array for a dot product
        labels = array(labels)
        # classify the data point, classification list is boolean
        data_lab = []
        for ii in data:
            data_lab.append(self.classify(ii))

        # Convert data from bool (0,1) to -1,+1
        data_lab = array(data_lab) * 2 - 1
        # return our correlation value which is the dot product of the
        # random labels and hypothesis predictions normalized by the data length
        cor_val = float(dot(data_lab, labels)) / float(len(data_lab))
        return cor_val


class PlaneHypothesis(Classifier):
    """
    A class that represents a decision boundary.
    """

    def __init__(self, x, y, b):
        """
        Provide the definition of the decision boundary's normal vector
        Args:
          x: First dimension
          y: Second dimension
          b: Bias term
        """
        self._vector = array([x, y])
        self._bias = b

    def __call__(self, point):
        return self._vector.dot(point) - self._bias

    def classify(self, point):
        return self(point) >= 0

    def __str__(self):
        return "x: x_0 * %0.2f + x_1 * %0.2f >= %f" % \
               (self._vector[0], self._vector[1], self._bias)


class OriginPlaneHypothesis(PlaneHypothesis):
    """
    A class that represents a decision boundary that must pass through the
    origin.
    """

    def __init__(self, x, y):
        """
        Create a decision boundary by specifying the normal vector to the
        decision plane.
        Args:
          x: First dimension
          y: Second dimension
        """
        PlaneHypothesis.__init__(self, x, y, 0)


class AxisAlignedRectangle(Classifier):
    """
    A class that represents a hypothesis where everything within a rectangle
    (inclusive of the boundary) is positive and everything else is negative.
    """

    def __init__(self, start_x, start_y, end_x, end_y):
        """
        Create an axis-aligned rectangle classifier.  Returns true for any
        points inside the rectangle (including the boundary)
        Args:
          start_x: Left position
          start_y: Bottom position
          end_x: Right position
          end_y: Top position
        """
        assert end_x >= start_x, "Cannot have negative length (%f vs. %f)" % \
                                 (end_x, start_x)
        assert end_y >= start_y, "Cannot have negative height (%f vs. %f)" % \
                                 (end_y, start_y)

        self._x1 = start_x
        self._y1 = start_y
        self._x2 = end_x
        self._y2 = end_y

    def classify(self, point):
        """
        Classify a data point
        Args:
          point: The point to classify
        """
        return (point[0] >= self._x1 and point[0] <= self._x2) and \
               (point[1] >= self._y1 and point[1] <= self._y2)

    def __str__(self):
        return "(%0.2f, %0.2f) -> (%0.2f, %0.2f)" % \
               (self._x1, self._y1, self._x2, self._y2)


class ConstantClassifier(Classifier):
    """
    A classifier that always returns true
    """

    def classify(self, point):
        return True


def constant_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over the single constant
    hypothesis possible.
    Args:
      dataset: The dataset to use to generate hypotheses
    """
    yield ConstantClassifier()


def origin_plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.
    Classifiers are represented as a vector.  The classification decision is
    the sign of the dot product between an input point and the classifier.
    Args:
      dataset: The dataset to use to generate hypotheses
    """
    # Machine precision
    eps = spacing(single(1))
    # Put the data into an numpy array
    data_array = array(dataset)
    # Break out X,Y vector components
    x_vec = data_array[:, 0]
    y_vec = data_array[:, 1]
    # Calculate Slopes of all the points
    m_vec = divide(y_vec, x_vec)
    # Sort the slope vector and remove duplicate entries
    m_vec = unique(sort(m_vec))
    # Initialize empty vector of interpolated boundaries for hypothesis 
    m_int_vec = empty(len(m_vec))
    for ii in range(0, len(m_vec) - 1):
        if (isinf(m_vec[ii + 1])):
            m_int_vec[ii] = 1. * m_vec[ii] + eps
        else:
            m_int_vec[ii] = 1. * (m_vec[ii + 1] + m_vec[ii]) / 2.

    # The largest slope will get increased by machine precision and will be our final boundary
    m_int_vec[len(m_vec) - 1] = m_vec[len(m_vec) - 1] + eps
    # Then make a vector of tuples to return that are inverted
    # to get the normal vector 
    yy = -1 * ones(len(m_int_vec))
    mm = empty([len(m_int_vec), 2])
    mm[:, 0] = m_int_vec
    mm[:, 1] = yy

    # Run a quick check for inf case
    for ii in range(len(mm)):
        if (isinf(mm[ii][0])):
            mm[ii][0] = 1.
            mm[ii][1] = eps

    ## Also append the inverse classifications
    mm = append(mm, -1 * mm, axis=0)
    for ii in mm:
        yield OriginPlaneHypothesis(ii[0], ii[1])


def check_intersection(hull1, hull2):
    bstX_h1 = BST()
    bstY_h1 = BST()

    maxX_h2 = float('-inf')
    minX_h2 = float('inf')
    maxY_h2 = float('-inf')
    minY_h2 = float('inf')
    ptArray = hull1.points
    ptArray2 = hull2.points
    for vertex in hull1.vertices:
        # print (ptArray[vertex])
        bstX_h1.insert(ptArray[vertex][0])
        bstY_h1.insert(ptArray[vertex][1])
    for vertex in hull2.vertices:
        maxX_h2 = max(maxX_h2, ptArray2[vertex][0])
        minX_h2 = min(minX_h2, ptArray2[vertex][0])
        maxY_h2 = max(maxY_h2, ptArray2[vertex][1])
        minY_h2 = min(minY_h2, ptArray2[vertex][1])

    len_x = functools.reduce(lambda x, y: x + 1, bstX_h1.range(minX_h2, maxX_h2), 0)
    len_y = functools.reduce(lambda x, y: x + 1, bstY_h1.range(minY_h2, maxY_h2), 0)
    return len_x > 0 and len_y > 0


def find_hull(c1, c2):
    hull_list1 = array([elem for elem in c1])
    hull_list2 = array([elem for elem in c2])
    h1 = []
    h2 = []
    hlen = len(hull_list1)
    for j in range(hlen):
        h1.append(ConvexHull(hull_list1[j]))
        h2.append(ConvexHull(hull_list2[hlen - j - 1]))

    return h1, h2


def find_hull_single(c1):
    hull_list1 = [elem for elem in c1]
    # print (hull_list1)
    # hull_list2=array([elem for elem in c2])
    h1 = []
    # h2=[]
    hlen = len(hull_list1)
    for j in range(hlen):
        h1.append(ConvexHull(hull_list1[j]))
        # h2.append(ConvexHull(hull_list2[hlen-j-1]))

    return h1


def point_in_hull(pt, h):
    v_list = [h.points[vertex] for vertex in h.vertices]
    maxX = max([v[0] for v in v_list])
    minX = min([v[0] for v in v_list])
    maxY = max([v[1] for v in v_list])
    minY = min([v[1] for v in v_list])
    return (pt[0] < maxX and pt[0] > minX) and (pt[1] < maxY and pt[1] > minY)


def closest_pair(l1, l2):
    min_dist = float('inf')
    min_l1_index = -1
    min_l2_index = -1
    for i in range(len(l1)):
        for j in range(len(l2)):
            dist = sqrt((l2[j][0] - l1[i][0]) ** 2 + (l2[j][1] - l1[i][1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                min_l1_index = i
                min_l2_index = j
    return min_l1_index, min_l2_index


def pbisect(hull1, hull2):
    ptsH1 = [hull1.points[vertex] for vertex in hull1.vertices]
    ptsH2 = [hull2.points[vertex] for vertex in hull2.vertices]
    # ptsH1=hull1
    # ptsH2=hull2
    idx1, idx2 = closest_pair(ptsH1, ptsH2)
    # print (ptsH1[idx1],ptsH2[idx2] )
    pt_slope = (ptsH2[idx2][1] - ptsH1[idx1][1]) / (ptsH2[idx2][0] - ptsH1[idx1][0])
    if pt_slope == 0:
        pt_slope += 1e-4
    mid_point = ((ptsH2[idx2][0] + ptsH1[idx1][0]) / 2, (ptsH2[idx2][1] + ptsH1[idx1][1]) / 2)
    # print (mid_point)
    # print (pt_slope)
    pbisect_slope = -1.0 / (pt_slope)
    # print (pbisect_slope)
    intercept = mid_point[1] - pbisect_slope * mid_point[0]
    return pbisect_slope, intercept


def find_mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def line_intersect(c1, c2):
    pt_slope = (c1[1][1] - c1[0][1]) / (c1[1][0] - c1[0][0])
    if pt_slope == 0:
        pt_slope += 1e-4
    pt_intercept = c1[1][1] - pt_slope * c1[1][0]
    tempa = c2[1][1] - pt_slope * c2[1][0] - pt_intercept
    tempb = c2[0][1] - pt_slope * c2[0][0] - pt_intercept
    return tempa * tempb < 0


def line_in_hull(c, h):
    pt_slope = (c[1][1] - c[0][1]) / (c[1][0] - c[0][0])
    if pt_slope == 0:
        pt_slope += 1e-4
    pt_intercept = c[1][1] - pt_slope * c[1][0]
    v_list = [h.points[vertex] for vertex in h.vertices]
    temp = [v[1] - pt_slope * v[0] - pt_intercept for v in v_list]
    temp = array(temp)
    return prod(temp) < 0


def plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector and a bias.  The classification
    decision is the sign of the dot product between an input point and the
    classifier plus a bias.

    Args:
      dataset: The dataset to use to generate hypotheses

    """
    hyp_list = []
    hyp_list_swap = []
    n = int(len(dataset))
    if n == 1:
        slope = random.randint(-10, 10)
        intercept = random.randint(-10, 10)
        hyp_list.append(PlaneHypothesis(-slope, 1, -intercept))
        hyp_list_swap.append(PlaneHypothesis(slope, -1, intercept))
    elif n == 2:
        pt_slope = (dataset[1][1] - dataset[0][1]) / (dataset[1][0] - dataset[0][0])
        if pt_slope == 0:
            pt_slope += 1e-4
        pt_intercept = dataset[1][1] - pt_slope * dataset[1][0]
        mid_point = ((dataset[1][0] + dataset[0][0]) / 2, (dataset[1][1] + dataset[0][1]) / 2)
        slope = -1.0 / (pt_slope)
        intercept = mid_point[1] - slope * mid_point[0]
        hyp_list.append(PlaneHypothesis(-slope, 1, -intercept))
        hyp_list_swap.append(PlaneHypothesis(slope, -1, intercept))
        hyp_list.append(PlaneHypothesis(-pt_slope, 1, -(intercept + 1e-16)))
        hyp_list.append(PlaneHypothesis(-pt_slope, 1, -(intercept - 1e-16)))

        # hyp_list_swap.append(PlaneHypothesis(slope,-1,intercept)))

    else:
        for i in range(0, int(n / 2) + 1):
            # print (i)
            # print (int(n/2))
            if i == 0:
                case = list(combinations(dataset, n))
                h1 = find_hull_single(case)
                # print len(h1))
                pt = h1[0].points[h1[0].vertices[0]]
                a, b, c = h1[0].equations[0]
                slope = float(b) / a
                intercept = pt[1] - slope * pt[0]
                hyp_list.append(PlaneHypothesis(-slope, 1, -intercept))
                hyp_list_swap.append(PlaneHypothesis(slope, -1, intercept))
            elif i == 1 and n == 3:
                case1 = list(combinations(dataset, i))
                case2 = list(combinations(dataset, n - i))
                for l in range(3):
                    mid_point = find_mid(case2[2-l][0], case2[2-l][1])
                    if (mid_point[0] - case1[l][0][0]) == 0:
                        den = (mid_point[0] - case1[l][0][0]) + 1e-5
                    else:
                        den = (mid_point[0] - case1[l][0][0])
                    pt_slope = (mid_point[1] - case1[l][0][1]) / den
                    if pt_slope == 0:
                        pt_slope += 1e-4
                    m_point = ((mid_point[0] + case1[l][0][0]) / 2, (mid_point[1] + case1[l][0][1]) / 2)
                    slope = -1.0 / (pt_slope)
                    intercept = m_point[1] - slope * m_point[0]
                    hyp_list.append(PlaneHypothesis(-slope, 1, -intercept))
                    hyp_list_swap.append(PlaneHypothesis(slope, -1, intercept))

            elif i == 1 and n > 3:
                case1 = list(combinations(dataset, i))
                case2 = list(combinations(dataset, n - i))
                h2 = find_hull_single(case2)
                for l in range(len(h2)):
                    if not point_in_hull(case1[l][0], h2[l]):
                        ptsH2 = [h2[l].points[vertex] for vertex in h2[l].vertices]
                        ptsH1 = [case1[l][0]]
                        idx1, idx2 = closest_pair(ptsH1, ptsH2)
                        pt_slope = (ptsH2[idx2][1] - ptsH1[idx1][1]) / (ptsH2[idx2][0] - ptsH1[idx1][0])
                        if pt_slope == 0:
                            pt_slope += 1e-4
                        mid_point = ((ptsH2[idx2][0] + ptsH1[idx1][0]) / 2, (ptsH2[idx2][1] + ptsH1[idx1][1]) / 2)
                        slope = -1.0 / (pt_slope)
                        intercept = mid_point[1] - slope * mid_point[0]
                        hyp_list.append(PlaneHypothesis(-slope, 1, -intercept))
                        hyp_list_swap.append(PlaneHypothesis(slope, -1, intercept))
            elif i == 2 and n == 4:
                case1 = list(combinations(dataset, i))
                case2 = list(combinations(dataset, n - i))
                case_len = len(case1)
                for l in range(case_len):
                    if not line_intersect(case1[l], case2[case_len - l - 1]):
                        ptsH2 = [case2[case_len - l - 1][0], case2[case_len - l - 1][1]]
                        ptsH1 = [case1[l][0], case1[l][1]]
                        # print(ptsH2,ptsH1)
                        idx1, idx2 = closest_pair(ptsH1, ptsH2)
                        pt_slope = (ptsH2[idx2][1] - ptsH1[idx1][1]) / (ptsH2[idx2][0] - ptsH1[idx1][0])
                        mid_point = ((ptsH2[idx2][0] + ptsH1[idx1][0]) / 2, (ptsH2[idx2][1] + ptsH1[idx1][1]) / 2)
                        if pt_slope == 0:
                            pt_slope += 1e-4
                        slope = -1.0 / (pt_slope)
                        intercept = mid_point[1] - slope * mid_point[0]
                        hyp_list.append(PlaneHypothesis(-slope, 1, -intercept))
                        hyp_list_swap.append(PlaneHypothesis(slope, -1, intercept))

            elif i == 2:
                case1 = list(combinations(dataset, i))
                case2 = list(combinations(dataset, n - i))
                case_len = len(case1)
                h2 = find_hull_single(case2)

                for l in range(case_len):
                    if not line_in_hull(case1[l], h2[l]):
                        ptsH2 = [h2[l].points[vertex] for vertex in h2[l].vertices]
                        ptsH1 = [case1[l][0], case1[l][1]]
                        idx1, idx2 = closest_pair(ptsH1, ptsH2)
                        pt_slope = (ptsH2[idx2][1] - ptsH1[idx1][1]) / (ptsH2[idx2][0] - ptsH1[idx1][0])
                        if pt_slope == 0:
                            pt_slope += 1e-4
                        mid_point = ((ptsH2[idx2][0] + ptsH1[idx1][0]) / 2, (ptsH2[idx2][1] + ptsH1[idx1][1]) / 2)
                        slope = -1.0 / (pt_slope)
                        intercept = mid_point[1] - slope * mid_point[0]
                        hyp_list.append(PlaneHypothesis(-slope, 1, -intercept))
                        hyp_list_swap.append(PlaneHypothesis(slope, -1, intercept))
            elif i > 2:
                case1 = list(combinations(dataset, i))
                case2 = list(combinations(dataset, n - i))

                h1, h2 = find_hull(case1, case2)  # Convex Hull Finder

                # Check Intersection
                Intersection_list = []
                for j in range(len(h1)):
                    Intersection_list.append(check_intersection(h1[j], h2[j]))

                k = 0
                for intersection in Intersection_list:
                    if not intersection:
                        slope, intercept = pbisect(h1[k], h2[k])
                        hyp_list.append(PlaneHypothesis(-slope, 1, -intercept))
                        hyp_list_swap.append(PlaneHypothesis(slope, -1, intercept))
                k += 1

    labelings = []
    final_classifiers = []
    for hh in hyp_list:
        pattern = [hh.classify(dataset[x]) for x in range(len(dataset))]
        present = pattern in labelings
        if not present:
            final_classifiers.append(hh)
            labelings.append(pattern)

    for hh in hyp_list_swap:
        pattern = [hh.classify(dataset[x]) for x in range(len(dataset))]
        present = pattern in labelings
        if not present:
            final_classifiers.append(hh)
            labelings.append(pattern)

    return final_classifiers


def make_a_rec(data_list, ind_list):
    eps = spacing(single(1))
    z = []
    for ii in ind_list:
        z.append(data_list[ii])

    z = array(z)
    x_vec = z[:, 0]
    y_vec = z[:, 1]
    x_min = min(x_vec) - eps
    y_min = min(y_vec) - eps
    x_max = max(x_vec) + eps
    y_max = max(y_vec) + eps
    min_max_ar = [x_min, y_min, x_max, y_max]
    min_max_ar = array(min_max_ar)
    return min_max_ar


# Make a null rectangle that yields all false values
# we do this by making an infentesmial box outside max(x),max(y),max(x),max(y) 
# nothing should fall in this box so we should get a false response.   
def make_null_rec(data_list):
    eps = spacing(single(1))
    z = []
    for ii in data_list:
        z.append(ii)

    z = array(z)
    x_vec = z[:, 0]
    y_vec = z[:, 1]
    x_min = max(x_vec) + eps
    y_min = max(y_vec) + eps
    x_max = max(x_vec) + 2 * eps
    y_max = max(y_vec) + 2 * eps
    min_max_ar = [x_min, y_min, x_max, y_max]
    min_max_ar = array(min_max_ar)
    return min_max_ar


# Check if one point is in the rectangle.
# candidate for BST acceleartion
# returns a boolean value True if it's in the rectangle, false if it's out
def check_a_point(rec_ar, data_point):
    dx = float(data_point[0])
    dy = float(data_point[1])
    x_min = rec_ar[0]
    y_min = rec_ar[1]
    x_max = rec_ar[2]
    y_max = rec_ar[3]
    return ((x_min < dx) and (dx < x_max) and (y_min < dy) and (dy < y_max))


# Check the whole data list by calling
# check_a_point for each data point
# returns a boolean classifier list    
def check_a_rec(rec_ar, data_list):
    bool_list = []
    for ii in data_list:
        bool_val = check_a_point(rec_ar, ii)
        bool_list.append(bool_val)

    return bool_list


# Checks a rectangle and boolean prediction hypothesis against a running
# dictionary of boolean predictions. If the hypothesis is new,
# the rectangle dictionary is updated along with the boolean hypothesis
# dictionary. The dictionaries are returned.    
def check_a_hyp(bool_list, rec_ar, rec_dict, hypo_dict):
    if not (bool_list in hypo_dict.values()):
        key_num = len(hypo_dict) + 1
        hypo_dict[key_num] = bool_list
        rec_dict[key_num] = rec_ar

    return hypo_dict, rec_dict


# This function generates all the index combinations
# for all possible non-zero lengths for a particular data length
# It returns a list of these values.                     
def itter_gen(data_len):
    it_lo = []
    dum_list = list(range(data_len))
    for xx in range(data_len):
        dum_gen = combinations(dum_list, xx + 1)
        for jj in dum_gen:
            it_lo.append(jj)

    return it_lo


# End axis_aligned_hypothese helper functions

def axis_aligned_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.
    Classifiers are axis-aligned rectangles
    Args:
      dataset: The dataset to use to generate hypotheses
    """

    # Machine precision value to give minimal headroom around rectangles
    eps = spacing(single(1))
    # Generate all the index combinations to consider for a Broot force solver
    ind_it = itter_gen(len(dataset))
    # print(ind_it)
    # Make empty dictionaries to hold our rectangle classifiers and their boolean hypotheses
    REC_DICT = defaultdict(array)
    HYP_DICT = defaultdict(list)
    # Iterate through all the hypotheses
    for ii in ind_it:
        # List of indecies to consider   
        ind_list = ii
        # Make a hypothesis rectangle from the list
        rec_hyp = make_a_rec(dataset, ind_list)
        # See how it classifies the points
        bool_out = check_a_rec(rec_hyp, dataset)
        # Check this rectangle hypothesis and its classification list agains the dictionaries
        # Update the dictionaries if this one isn't in there yet
        HYP_DICT, REC_DICT = check_a_hyp(bool_out, rec_hyp, REC_DICT, HYP_DICT)

    # The last thing to do is generate a null-hypothesis rectangle
    # This wouldn't be generate from the points by themselves as 
    # those steps always result in at least one inclusion
    null_rec = make_null_rec(dataset)
    bool_out = check_a_rec(null_rec, dataset)
    HYP_DICT, REC_DICT = check_a_hyp(bool_out, null_rec, REC_DICT, HYP_DICT)
    rec_vals = REC_DICT.values()
    for ii in rec_vals:
        yield AxisAlignedRectangle(float(ii[0]), float(ii[1]), float(ii[2]), float(ii[3]))


def coin_tosses(number, random_seed=0):
    """
    Generate a desired number of coin tosses with +1/-1 outcomes.
    Args:
      number: The number of coin tosses to perform
      random_seed: The random seed to use
    """
    if random_seed != 0:
        seed(random_seed)

    return [randint(0, 1) * 2 - 1 for x in range(number)]


def rademacher_estimate(dataset, hypothesis_generator, num_samples=500,
                        random_seed=0):
    """
    Given a dataset, estimate the rademacher complexity
    Args:
      dataset: a sequence of examples that can be handled by the hypotheses
      generated by the hypothesis_generator
      hypothesis_generator: a function that generates an iterator over
      hypotheses given a dataset
      num_samples: the number of samples to use in estimating the Rademacher
      correlation
    """
    # Make an empty vector to hold the correlation results for each step to take the mean 
    cor_mean_vec = []
    # Iterate over num_samples to get an expected value
    for hh in range(num_samples):
        # Make some random values
        if random_seed != 0:
            rand_lab = coin_tosses(len(dataset), random_seed + hh)
        else:
            rand_lab = coin_tosses(len(dataset))

        # Make sure to regenerate the hypothesis set EACH time!
        h_gen = hypothesis_generator(dataset)
        # Make a vector to hold the correlation results for our set of hypotheses
        dat_cor = []
        # Loop through all hypotheses and save the correlation results for each one
        for ii in h_gen:
            dat_cor.append(ii.correlation(dataset, rand_lab))

        # Append the max result of looping through one hypotheses set, will end up being num_samples long
        cor_mean_vec.append(max(dat_cor))

    # To finish the rademacher estimate, take the expected value of our runs
    mean_out = mean(array(cor_mean_vec))
    return mean_out


if __name__ == "__main__":
    print("Rademacher correlation of constant classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, constant_hypotheses))
    print("Rademacher correlation of rectangle classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, axis_aligned_hypotheses))
    print("Rademacher correlation of origin plane classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, origin_plane_hypotheses))
    print("Rademacher correlation of plane classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, plane_hypotheses))
