# Solve Sudoku puzzles using AC-3 and Backtracking algorithms
'''
YOUR TASKS:
1. Read to understand the following code
2. Implement the AC3() function
3. Give comments on the backtracking_search() function to show your comprehensive understanding of the code
4. (Optional) Add GUI, animation...
5. Input a Sudoku to try your code
'''

# Source code của Thầy
import sys
import itertools
import re
import random
from functools import reduce
import numpy as np

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# %% Utilities
from itertools import count


def first(iterable, default=None):
    """Return the first element of an iterable; or default."""
    return next(iter(iterable), default)


def count(seq):
    """Count the number of items in sequence that are interpreted as true."""
    return sum(map(bool, seq))


def argmin_random_tie(seq, key=lambda x: x):
    """Return a minimum element of seq; break ties at random."""
    items = list(seq)
    random.shuffle(items)  # Randomly shuffle a copy of seq.
    return min(items, key=key)


def flatten(seqs):
    return sum(seqs, [])


def different_values_constraint(A, a, B, b):
    """A constraint saying two neighboring variables must differ in value."""
    return a != b


# %% CSP
class CSP():
    """This class describes finite-domain Constraint Satisfaction Problems.
    A CSP is specified by the following inputs:
        variables   A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b
    """

    def __init__(self, variables, domains, neighbors, constraints):
        """Construct a CSP problem. If variables is empty, it becomes domains.keys()."""
        variables = variables or list(domains.keys())
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.curr_domains = None
        self.nassigns = 0

    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""
        assignment[var] = val
        self.nassigns += 1
        return assignment

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        # Subclasses may implement this more efficiently
        def conflict(var2):
            return var2 in assignment and not self.constraints(var, val, var2, assignment[var2])

        return count(conflict(v) for v in self.neighbors[var])


# %% Sudoku problem
# Constants and delarations to display and work with Sudoku grid
_R3 = list(range(3))
_CELL = itertools.count().__next__
_BGRID = [[[[_CELL() for x in _R3] for y in _R3] for bx in _R3] for by in _R3]
_BOXES = flatten([list(map(flatten, brow)) for brow in _BGRID])
_ROWS = flatten([list(map(flatten, zip(*brow))) for brow in _BGRID])
_COLS = list(zip(*_ROWS))
_NEIGHBORS = {v: set() for v in flatten(_ROWS)}
for unit in map(set, _BOXES + _ROWS + _COLS):
    for v in unit:
        _NEIGHBORS[v].update(unit - {v})


class Sudoku(CSP):
    #
    """
    A Sudoku problem.
    init_assignment is a string of 81 digits for 81 cells, row by row.
    Each filled cell holds a digit in 1..9. Each empty cell holds 0 or '.'

    For example, for the below Sodoku
    >>> init_assignment = '..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..'

    >>> sodoku1 = Sudoku(init_assignment)
    >>> sodoku1.display()
    . . 3 | . 2 . | 6 . .
    9 . . | 3 . 5 | . . 1
    . . 1 | 8 . 6 | 4 . .
    ------+-------+------
    . . 8 | 1 . 2 | 9 . .
    7 . . | . . . | . . 8
    . . 6 | 7 . 8 | 2 . .
    ------+-------+------
    . . 2 | 6 . 9 | 5 . .
    8 . . | 2 . 3 | . . 9
    . . 5 | . 1 . | 3 . .

    >>> sodoku1.display_variables()
      0  1  2  |  9 10 11  | 18 19 20
      3  4  5  | 12 13 14  | 21 22 23
      6  7  8  | 15 16 17  | 24 25 26
     --------------------------------
     27 28 29  | 36 37 38  | 45 46 47
     30 31 32  | 39 40 41  | 48 49 50
     33 34 35  | 42 43 44  | 51 52 53
     --------------------------------
     54 55 56  | 63 64 65  | 72 73 74
     57 58 59  | 66 67 68  | 75 76 77
     60 61 62  | 69 70 71  | 78 79 80

    >>> AC3(sodoku1)
    True
    >>> sodoku1.display()
    4 8 3 | 9 2 1 | 6 5 7
    9 6 7 | 3 4 5 | 8 2 1
    2 5 1 | 8 7 6 | 4 9 3
    ------+-------+------
    5 4 8 | 1 3 2 | 9 7 6
    7 2 9 | 5 6 4 | 1 3 8
    1 3 6 | 7 9 8 | 2 4 5
    ------+-------+------
    3 7 2 | 6 8 9 | 5 1 4
    8 1 4 | 2 5 3 | 7 6 9
    6 9 5 | 4 1 7 | 3 8 2
    """

    R3 = _R3
    Cell = _CELL
    bgrid = _BGRID
    boxes = _BOXES
    rows = _ROWS
    cols = _COLS
    neighbors = _NEIGHBORS

    def __init__(self, grid):
        """Build a Sudoku problem from a string representing the grid:
        the digits 1-9 denote a filled cell, '.' or '0' an empty one;
        other characters are ignored."""

        squares = re.findall(r'\d|\.', grid)

        # NOTE: For variables in order of in order of 3x3 BOXES:
        domains = {var: list(ch) if ch in '123456789' else list('123456789')
                   for var, ch in zip(flatten(self.rows), squares)}  #

        if len(squares) > 81:
            raise ValueError("Not a Sudoku grid", grid)  # Too many squares

        # For variables in order of in order of 3x3 BOXES:
        neighbors = {0: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 27, 30, 33, 54, 57, 60},
                     1: {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 28, 31, 34, 55, 58, 61},
                     2: {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 29, 32, 35, 56, 59, 62},
                     9: {0, 1, 2, 66, 69, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 36, 39, 42, 63},
                     10: {0, 1, 2, 64, 67, 70, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 37, 40, 43},
                     11: {0, 1, 2, 65, 68, 71, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 38, 41, 44},
                     18: {0, 1, 2, 72, 9, 10, 11, 75, 78, 19, 20, 21, 22, 23, 24, 25, 26, 45, 48, 51},
                     19: {0, 1, 2, 9, 10, 11, 73, 76, 79, 18, 20, 21, 22, 23, 24, 25, 26, 46, 49, 52},
                     20: {0, 1, 2, 9, 10, 11, 74, 77, 80, 18, 19, 21, 22, 23, 24, 25, 26, 47, 50, 53},
                     3: {0, 1, 2, 4, 5, 6, 7, 8, 12, 13, 14, 21, 22, 23, 27, 30, 33, 54, 57, 60},
                     4: {0, 1, 2, 3, 5, 6, 7, 8, 12, 13, 14, 21, 22, 23, 28, 31, 34, 55, 58, 61},
                     5: {0, 1, 2, 3, 4, 6, 7, 8, 12, 13, 14, 21, 22, 23, 29, 32, 35, 56, 59, 62},
                     12: {66, 3, 4, 5, 69, 9, 10, 11, 13, 14, 15, 16, 17, 21, 22, 23, 36, 39, 42, 63},
                     13: {64, 3, 4, 5, 67, 70, 9, 10, 11, 12, 14, 15, 16, 17, 21, 22, 23, 37, 40, 43},
                     14: {65, 3, 4, 5, 68, 71, 9, 10, 11, 12, 13, 15, 16, 17, 21, 22, 23, 38, 41, 44},
                     21: {3, 4, 5, 72, 75, 12, 13, 14, 78, 18, 19, 20, 22, 23, 24, 25, 26, 45, 48, 51},
                     22: {3, 4, 5, 73, 12, 13, 14, 76, 79, 18, 19, 20, 21, 23, 24, 25, 26, 46, 49, 52},
                     23: {3, 4, 5, 74, 12, 13, 14, 77, 80, 18, 19, 20, 21, 22, 24, 25, 26, 47, 50, 53},
                     6: {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 24, 25, 26, 27, 30, 33, 54, 57, 60},
                     7: {0, 1, 2, 3, 4, 5, 6, 8, 15, 16, 17, 24, 25, 26, 28, 31, 34, 55, 58, 61},
                     8: {0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 24, 25, 26, 29, 32, 35, 56, 59, 62},
                     15: {66, 69, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 24, 25, 26, 36, 39, 42, 63},
                     16: {64, 67, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 24, 25, 26, 70, 37, 40, 43},
                     17: {65, 68, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 25, 26, 38, 71, 41, 44},
                     24: {6, 7, 8, 72, 75, 78, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 45, 48, 51},
                     25: {6, 7, 8, 73, 76, 79, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 46, 49, 52},
                     26: {6, 7, 8, 74, 77, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 80, 47, 50, 53},
                     27: {0, 3, 6, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 45, 46, 47, 54, 57, 60},
                     28: {1, 4, 7, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 45, 46, 47, 55, 58, 61},
                     29: {2, 5, 8, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 45, 46, 47, 56, 59, 62},
                     36: {66, 69, 9, 12, 15, 27, 28, 29, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 63},
                     37: {64, 67, 70, 10, 13, 16, 27, 28, 29, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47},
                     38: {65, 68, 71, 11, 14, 17, 27, 28, 29, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47},
                     45: {72, 75, 78, 18, 21, 24, 27, 28, 29, 36, 37, 38, 46, 47, 48, 49, 50, 51, 52, 53},
                     46: {73, 76, 79, 19, 22, 25, 27, 28, 29, 36, 37, 38, 45, 47, 48, 49, 50, 51, 52, 53},
                     47: {74, 77, 80, 20, 23, 26, 27, 28, 29, 36, 37, 38, 45, 46, 48, 49, 50, 51, 52, 53},
                     30: {0, 3, 6, 27, 28, 29, 31, 32, 33, 34, 35, 39, 40, 41, 48, 49, 50, 54, 57, 60},
                     31: {1, 4, 7, 27, 28, 29, 30, 32, 33, 34, 35, 39, 40, 41, 48, 49, 50, 55, 58, 61},
                     32: {2, 5, 8, 27, 28, 29, 30, 31, 33, 34, 35, 39, 40, 41, 48, 49, 50, 56, 59, 62},
                     39: {66, 69, 9, 12, 15, 30, 31, 32, 36, 37, 38, 40, 41, 42, 43, 44, 48, 49, 50, 63},
                     40: {64, 67, 70, 10, 13, 16, 30, 31, 32, 36, 37, 38, 39, 41, 42, 43, 44, 48, 49, 50},
                     41: {65, 68, 71, 11, 14, 17, 30, 31, 32, 36, 37, 38, 39, 40, 42, 43, 44, 48, 49, 50},
                     48: {72, 75, 78, 18, 21, 24, 30, 31, 32, 39, 40, 41, 45, 46, 47, 49, 50, 51, 52, 53},
                     49: {73, 76, 79, 19, 22, 25, 30, 31, 32, 39, 40, 41, 45, 46, 47, 48, 50, 51, 52, 53},
                     50: {74, 77, 80, 20, 23, 26, 30, 31, 32, 39, 40, 41, 45, 46, 47, 48, 49, 51, 52, 53},
                     33: {0, 3, 6, 27, 28, 29, 30, 31, 32, 34, 35, 42, 43, 44, 51, 52, 53, 54, 57, 60},
                     34: {1, 4, 7, 27, 28, 29, 30, 31, 32, 33, 35, 42, 43, 44, 51, 52, 53, 55, 58, 61},
                     35: {2, 5, 8, 27, 28, 29, 30, 31, 32, 33, 34, 42, 43, 44, 51, 52, 53, 56, 59, 62},
                     42: {66, 69, 9, 12, 15, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 51, 52, 53, 63},
                     43: {64, 67, 70, 10, 13, 16, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 51, 52, 53},
                     44: {65, 68, 71, 11, 14, 17, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 51, 52, 53},
                     51: {72, 75, 78, 18, 21, 24, 33, 34, 35, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53},
                     52: {73, 76, 79, 19, 22, 25, 33, 34, 35, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53},
                     53: {74, 77, 80, 20, 23, 26, 33, 34, 35, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52},
                     54: {64, 65, 0, 3, 6, 72, 73, 74, 27, 30, 33, 55, 56, 57, 58, 59, 60, 61, 62, 63},
                     55: {64, 65, 1, 4, 7, 72, 73, 74, 28, 31, 34, 54, 56, 57, 58, 59, 60, 61, 62, 63},
                     56: {64, 65, 2, 5, 72, 73, 74, 8, 29, 32, 35, 54, 55, 57, 58, 59, 60, 61, 62, 63},
                     63: {64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 9, 12, 15, 36, 39, 42, 54, 55, 56},
                     64: {65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 10, 13, 16, 37, 40, 43, 54, 55, 56, 63},
                     65: {64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 11, 14, 17, 38, 41, 44, 54, 55, 56, 63},
                     72: {64, 65, 73, 74, 75, 76, 77, 78, 79, 80, 18, 21, 24, 45, 48, 51, 54, 55, 56, 63},
                     73: {64, 65, 72, 74, 75, 76, 77, 78, 79, 80, 19, 22, 25, 46, 49, 52, 54, 55, 56, 63},
                     74: {64, 65, 72, 73, 75, 76, 77, 78, 79, 80, 20, 23, 26, 47, 50, 53, 54, 55, 56, 63},
                     57: {0, 66, 67, 68, 3, 6, 75, 76, 77, 27, 30, 33, 54, 55, 56, 58, 59, 60, 61, 62},
                     58: {1, 66, 67, 68, 4, 7, 75, 76, 77, 28, 31, 34, 54, 55, 56, 57, 59, 60, 61, 62},
                     59: {66, 67, 68, 2, 5, 8, 75, 76, 77, 29, 32, 35, 54, 55, 56, 57, 58, 60, 61, 62},
                     66: {64, 65, 67, 68, 69, 70, 71, 9, 75, 76, 77, 12, 15, 36, 39, 42, 57, 58, 59, 63},
                     67: {64, 65, 66, 68, 69, 70, 71, 10, 75, 76, 77, 13, 16, 37, 40, 43, 57, 58, 59, 63},
                     68: {64, 65, 66, 67, 69, 70, 71, 75, 76, 77, 11, 14, 17, 38, 41, 44, 57, 58, 59, 63},
                     75: {66, 67, 68, 72, 73, 74, 76, 77, 78, 79, 80, 18, 21, 24, 45, 48, 51, 57, 58, 59},
                     76: {66, 67, 68, 72, 73, 74, 75, 77, 78, 79, 80, 19, 22, 25, 46, 49, 52, 57, 58, 59},
                     77: {66, 67, 68, 72, 73, 74, 75, 76, 78, 79, 80, 20, 23, 26, 47, 50, 53, 57, 58, 59},
                     60: {0, 3, 69, 70, 71, 6, 78, 79, 80, 27, 30, 33, 54, 55, 56, 57, 58, 59, 61, 62},
                     61: {1, 4, 69, 70, 71, 7, 78, 79, 80, 28, 31, 34, 54, 55, 56, 57, 58, 59, 60, 62},
                     62: {2, 69, 70, 71, 5, 8, 78, 79, 80, 29, 32, 35, 54, 55, 56, 57, 58, 59, 60, 61},
                     69: {64, 65, 66, 67, 68, 70, 71, 9, 12, 78, 79, 80, 15, 36, 39, 42, 60, 61, 62, 63},
                     70: {64, 65, 66, 67, 68, 69, 71, 10, 13, 78, 79, 80, 16, 37, 40, 43, 60, 61, 62, 63},
                     71: {64, 65, 66, 67, 68, 69, 70, 11, 78, 79, 80, 14, 17, 38, 41, 44, 60, 61, 62, 63},
                     78: {69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 18, 21, 24, 45, 48, 51, 60, 61, 62},
                     79: {69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 19, 22, 25, 46, 49, 52, 60, 61, 62},
                     80: {69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 20, 23, 26, 47, 50, 53, 60, 61, 62}}

        CSP.__init__(self, list(domains.keys()), domains, neighbors, different_values_constraint)

    def display(self):  # For variables in order of in order of 3x3 BOXES
        """Show a human-readable representation of the Sudoku."""
        place = 0
        if self.curr_domains is not None:
            self.domains = self.curr_domains.copy()
        for var in self.domains.keys():
            if place % 3 == 0 and place % 9 != 0:
                print(' |', end='')
            if place % 9 == 0 and place != 0:
                print('')
            if place % 27 == 0 and place != 0:
                print('------+-------+------')

            if len(self.domains[var]) == 1:
                if place % 9 == 0:
                    # Check các vị trí xuống hàng để sao cho board giống như mong đợi
                    print('%1s' % self.domains[var][0], end='')
                else:
                    print('%2s' % self.domains[var][0], end='')
            else:
                if place % 9 == 0:
                    # Check các vị trí xuống hàng để sao cho board giống như mong đợi
                    print('.', end='')
                else:
                    print(' .', end='')
            place += 1

    def display_variables(self):  # For variables in order of in order of 3x3 BOXES
        place = 0
        for var in self.domains.keys():
            if place % 3 == 0 and place % 9 != 0:
                print('  |', end='')
            if place % 9 == 0 and place != 0:
                print('')
            if place % 27 == 0 and place != 0:
                print(' --------------------------------')

            print('%3s' % var, end='')

            place += 1


# %%  Constraint Propagation with AC3
''' IMPLEMENT THE FOLLOWING FUNCTION '''


def AC3(csp):
    """See [Figure 6.3] for the algorithm"""
    queue = {(Xi, Xj) for Xi in csp.variables for Xj in csp.neighbors[Xi]}
    # add Xi (local) and Xj (neighbors) to queue
    csp.curr_domains = csp.domains.copy()  # curr_domains: a copy of domains. We will do inference on curr_domains

    ''' ADD YOUR CODE HERE '''
    while queue:
        (Xi, Xj) = queue.pop()
        # Lấy biến Xi và Xj làm tiền đề cho hàm revise để thu hẹp domain value
        if revise(csp, Xi, Xj):
            '''
            revise function giúp thúc đẩy domain value sao cho thỏa điều kiện
            ràng buộc giữa hai biến gần nhau x1 -> x2 (arc consistency)
            '''
            if len(csp.curr_domains) == 0:
                return False
            for Xk in csp.neighbors[Xi]:
                queue.add((Xk, Xi))

    return True  # CSP is satisfiable


def revise(csp, Xi, Xj):
    """Return true if we remove a value."""
    revised = False
    for x in csp.domains[Xi]:
        conflict = True

        for y in csp.curr_domains[Xj]:
            if csp.constraints(Xi, x, Xj, y):
                conflict = False
                break

        if conflict:
            csp.domains[Xi].remove(x)
            revised = True

    return revised


# %%  CSP Backtracking Search
# Variable ordering
def first_unassigned_variable(assignment, csp):  # random selection
    """The default variable order."""
    return first([var for var in csp.variables if var not in assignment])


def num_legal_values(csp, var, assignment):
    if csp.curr_domains:
        return len(csp.curr_domains[var])
    else:
        return count(csp.nconflicts(var, val, assignment) == 0 for val in csp.domains[var])


def minimum_remaining_values(assignment, csp):
    """Minimum-remaining-values heuristic."""
    return argmin_random_tie([v for v in csp.variables if v not in assignment],
                             key=lambda var: num_legal_values(csp, var, assignment))


# Value ordering
def unordered_domain_values(var, assignment, csp):  # random selection
    """The default value order."""
    return (csp.curr_domains or csp.domains)[var]


def least_constraining_value(var, assignment, csp):
    """Least-constraining-values heuristic."""
    return sorted((csp.curr_domains or csp.domains)[var], key=lambda val: csp.nconflicts(var, val, assignment))


# Inference
def forward_checking(csp, var, value, assignment, removals):
    """Prune neighbor values inconsistent with var=value."""
    for B in csp.neighbors[var]:
        if B not in assignment:
            for b in csp.curr_domains[B][:]:
                if not csp.constraints(var, value, B, b):
                    csp.curr_domains[B].remove(b)
                    if removals is not None:
                        removals.append((B, b))  # variable B and value b are removed from its domain
            if not csp.curr_domains[B]:
                return False
    return True


''' READ AND COMMENT to show your comprehensive understanding of the following function '''


# Backtracking search
# Bai tap 2
def backtracking_search(csp, select_unassigned_variable=minimum_remaining_values,
                        order_domain_values=least_constraining_value,
                        inference=forward_checking):  # Đặt lại tên cho function
    """See [Figure 6.5] for the algorithm"""

    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            # Stop condition if the number of variable assigned value = number of variable
            # (number of cell unassigned) -> return assignment (result)

            return assignment
        var = select_unassigned_variable(assignment, csp)
        # Choose a variable that is unassigned value

        for value in order_domain_values(var, assignment, csp):
            # Lấy giá trị trong miền để xét xem ô đó có giá trị bị trùng hay không.

            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                # Nếu nconficts == 0 (không trùng) thì ta gán giá trị đó vào biến

                removals = [(var, a) for a in csp.curr_domains[var] if a != value]
                # Sau đó, ta lấy giá trị của biến được gán (var, a) nếu biến chưa có
                # trong curr_domains (curr_domains is a copy of domains)
                # là tiền để cho hàm inference

                csp.curr_domains[var] = [value]
                # Tiếp theo ta gán giá trị của biến vừa gán vào domain tạm thời

                if inference(csp, var, value, assignment, removals):
                    # Kiểm tra domain của các biến liên quan xem có thây đổi không
                    # nếu domain chỉ còn 1 giá trị thì ta sẻ gán vào biến
                    result = backtrack(assignment)

                    #
                    if result is not None:
                        # if result = None -> solution
                        return result

                restore(csp, removals)
                # Thêm thêm giá trị removals vào curr_domains
                # removals chứa vị trí của biến và giá trị của biến đó
                # Thêm giá b vào vị trí B trong curr_domain

        csp.unassign(var, assignment)
        # Nếu giá trị bị trùng thì ta sẻ bỏ gán
        # print("Running")
        return None

    csp.curr_domains = csp.domains.copy()  # copy domains vào curr_domains
    result = backtrack({})
    # ta sẻ dừng backtrack cho tới khi tìm được solution
    return result


def restore(csp, removals):
    """Undo a supposition and all inferences from it."""
    for B, b in removals:
        csp.curr_domains[B].append(b)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(100, 100, 1200, 850)
        self.solution = []
        self.label_loop = QLabel("", self)
        self.init = []
        self.board = []

    def log(self):
        print()

    # init Gui when start game
    def game_play(self, board, init):
        self.board = board
        self.init = init
        self.reset()

    def reset(self):
        self.hide()
        label = QLabel("Wellcome guy", self)
        label.setGeometry(850, 20, 200, 50)
        label.setFont(QFont('Roboto', 12))
        button_play = QPushButton("Play game", self)
        button_play.setGeometry(850, 80, 200, 50)
        button_play.clicked.connect(self.show_board)
        button_play = QPushButton("Reset", self)
        button_play.setGeometry(850, 132, 200, 50)
        button_play.clicked.connect(self.reset)
        y_axis = -30
        for x in range(9):
            x_axis = 20
            y_axis += 80
            for y in range(9):
                if self.init[x][y] == 0:
                    button = QPushButton("", self)
                    button.setGeometry(x_axis, y_axis, 83, 83)
                    button.setStyleSheet(u"background-color: #ECF0F1")

                else:
                    button = QPushButton(str(self.init[x][y]), self)
                    button.setGeometry(x_axis, y_axis, 83, 83)
                    button.setStyleSheet(u"background-color: lightblue")
                    button.setFont(QFont("Roboto", 17))
                x_axis += 80
        self.show()

    def show_board(self):
        self.hide()
        y_axis = -30
        for x in range(9):
            x_axis = 20
            y_axis += 80
            for y in range(9):
                if self.board[x][y] == 0:
                    button = QPushButton("", self)
                    button.setGeometry(x_axis, y_axis, 83, 83)
                else:
                    button = QPushButton(str(self.board[x][y]), self)
                    button.setGeometry(x_axis, y_axis, 83, 83)
                    button.setStyleSheet(u"background-color: #ECF0F1; color: #5D6D7E")

                    button.setFont(QFont("Roboto", 17))
                x_axis += 80
        self.show()


def de_init_board(init_assign_hard):
    init_board = []
    for i in init_assign_hard:
        if i == ".":
            init_board.append(0)
        else:
            init_board.append(int(i))
    board = []
    board.append(init_board[0:9])
    board.append(init_board[9:18])
    board.append(init_board[18:27])
    board.append(init_board[27:36])
    board.append(init_board[36:45])
    board.append(init_board[45:54])
    board.append(init_board[54:63])
    board.append(init_board[63:72])
    board.append(init_board[72:81])
    return board


def boards(domains):
    de_board = []
    for i in domains:
        de_board.append(domains[i])
    # print("\n", de_board)
    a = []
    for i in range(len(de_board)):
        a.append(int(de_board[i][0]))
    print()
    # print(a[0:9])
    # print(a[9:18])
    # print(a[18:27])
    # print(a[27:36])
    # print(a[36:45])
    # print(a[45:54])
    # print(a[54:63])
    # print(a[63:72])
    # print(a[72:81])
    board = []
    board.append(a[0:9])
    board.append(a[9:18])
    board.append(a[18:27])
    board.append(a[27:36])
    board.append(a[36:45])
    board.append(a[45:54])
    board.append(a[54:63])
    board.append(a[63:72])
    board.append(a[72:81])
    print(board)
    return board


# %% main
if __name__ == '__main__':
    init_assign_easy = '..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..'
    sudoku = Sudoku(init_assign_easy)
    print('VARIABLES:')
    sudoku.display_variables()
    print('\nEASY SUDOKU:')
    sudoku.display()
    AC3(sudoku)
    print('\nSOLUTION TO THE EASY SUDOKU:')
    sudoku.display()

    # The init data hard sudoku https://sudoku.com/hard/
    init_assign_hard = '.........6...3...4..25....83....58.77..1.....8..2....6..6..1.4...94....5......17.'
    # The init data expert sudoku https://sudoku.com/expert/
    init_assign_expert = '.....4..7.2..1.95.6..2.......68...1..5.1..4.82.....6......7....18.........4.6..9.'
    # The init data expert sudoku https://sudoku.com/evil/
    init_assign_evil = '..4......2.5..18.....8....3.9...........7..6.1.8..53...3...9....4....2..9.2.5...7'
    sudoku1 = Sudoku(init_assign_evil)
    # get init board
    init = de_init_board(init_assign_evil)
    print("\n", init)
    print('\nHARD SUDOKU:')
    # sudoku.display()
    backtracking_search(sudoku1)
    print('\nSOLUTION TO THE HARD SUDOKU:')
    # sudoku.display()
    # print("\n", sudoku.domains)
    # boards(sudoku.domains)

    # GUI
    # Em sẽ sử dụng lại giao diện của các tuần trước
    # get solution board
    solution = boards(sudoku1.domains)
    app = QApplication(sys.argv)
    window = MainWindow()

    window.game_play(solution, init)
    sys.exit(app.exec())





