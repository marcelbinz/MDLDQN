import csv
import numpy as np

'''
    % Data:
    % Each csv file contains data for a single experiment.
    % - data1.csv is a bandit task with one stochastic arm and one deterministic arm
    % - data2.csv is a bandit task with two stochastic arms
    %
    % Columns in each csv file:
    % 1) subject #
    % 2) block #
    % 3) trial #
    % 4) mu1 (mean reward for arm 1)
    % 5) mu2 (mean reward for arm 2)
    % 6) choice (which arm was selected)
    % 7) reward (points)
    % 8) response time (milliseconds)
    %
    % OUTPUTS:
    % data - structure with the following fields:
    %   .N - # of trials
    %   .C - # of options
    %   .R - [N x 2] mean rewards for each arm
    %   .block - [N x 1] block #
    %   .trial - [N x 1] trial #
    %   .c - [N x 1] choice
    %   .rt - [N x 1] response time
    %
    % Sam Gershman, Oct 2017
'''

def load_csv(filename):
    with open(filename) as f:
        lines = csv.reader(f, delimiter=',')

        # don't read header
        M = np.zeros((sum(1 for l in open(filename))-1, 8))

        # put into matrix
        for i, l in enumerate(lines):
            if i == 0:
                continue
            M[i-1, :] = np.array(l).astype('float')
        S = np.unique(M[:, 0])

        d = {}
        for s in range(S.shape[0]):
            ix = np.logical_and(M[:, 0] == S[s], M[:, 7] < 20000) # exclude trials on which subjects took longer than 20 seconds to respond
            d[0, s] = {}
            d[0, s]['R'] = M[ix, 3:5]
            d[0, s]['block'] = M[ix, 1]
            d[0, s]['c'] = np.expand_dims(M[ix, 5], axis=1).astype('int')
            d[0, s]['r'] = M[ix, 6]
            d[0, s]['rt'] = M[ix, 7]
            d[0, s]['trial'] = M[ix, 2]
            d[0, s]['N'] = np.sum(ix)
            d[0, s]['C'] = 2

        return d
    return None

def kalman_filter(param, data):
    l = np.empty((data['block'].shape[0], 2), dtype=[('m', np.float64), ('s', np.float64), ('v', 'O'), ('p', 'O')])
    q = param[0]
    q1 = param[1]
    q2 = param[2]
    m = np.zeros((1, 2)).astype('float')
    s = np.array([[q1, q2]]).astype('float')

    for i in range(data['block'].shape[0]):
        if i == 0 or data['block'][i] != data['block'][i-1]:
            m = np.zeros((1, 2)).astype('float')
            s = np.array([[q1, q2]]).astype('float')
        c = data['c'][i, 0] - 1
        r = data['r'][i]

        # store latents
        l['m'][i, :] = m
        l['s'][i, :] = s

        # update
        k = s[0, c] / (s[0, c]+q)
        err = r - m[0, c]
        m[0, c] = m[0, c] + k*err
        s[0, c] = s[0, c] - k*s[0, c]

    return l
