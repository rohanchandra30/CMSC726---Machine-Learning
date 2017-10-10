from numpy import array, zeros, dot

kINSP = array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])


def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector.
    """

    w = zeros(len(x[0]))

    for ii in range(len(y)):
        w[0] = w[0] + alpha[ii] * y[ii] * x[ii][0]
        w[1] = w[1] + alpha[ii] * y[ii] * x[ii][1]

    return w


def find_support(x, y, w, b, tolerance=0.001):
    """
    Given a primal support vector, return the indices for all of the support
    vectors
    """

    min_r = 1.-tolerance
    max_r = 1.+tolerance
    support = set()
    for ii in range(len(y)):
        print(y[ii],dot(w,x[ii])+b)
        test_val = y[ii]*(dot(w,x[ii])+b)
        #print test_val
        if ((test_val>min_r) and (test_val<max_r)):
            support.add(ii)
            print(x[ii])

    return support

def find_slack(x, y, w, b):
    """
    Given a primal support vector instance, return the indices for all of the
    slack vectors
    """
    slack = set()
    for ii in range(len(y)):
        test_val = y[ii] * (dot(w, x[ii]) + b)
        # print test_val
        if (test_val < 1.0):
            slack.add(ii)

    return slack
