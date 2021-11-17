# Solve N-queens problem using Min-conflicts algorithm
'''
YOUR TASKS:
1. Read to understand the following code 
2. Give comments on the min_conflicts() function to show your comprehensive understanding of the code
3. (Optional) Add GUI, animation...
'''

import random

#%% Utilities:
from itertools import count, chain
import numpy as np
import sys
from timeit import  default_timer as timer
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
def argmin_random_tie(seq, key=lambda x: x):
    """Return a minimum element of seq; break ties at random."""
    items = list(seq)
    random.shuffle(items) #Randomly shuffle a copy of seq.
    return min(items, key=key)

class UniversalDict:
    """A universal dict maps any key to the same value. We use it here
    as the domains dict for CSPs in which all variables have the same domain.
    >>> d = UniversalDict(42)
    >>> d['life']
    42
    """      
    def __init__(self, value): self.value = value

    def __getitem__(self, key): return self.value

    def __repr__(self): return '{{Any: {0!r}}}'.format(self.value)


#%% CSP
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
        #super().__init__(())
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

    # This is for min_conflicts search  
    def conflicted_vars(self, current):
        """Return a list of variables in current assignment that are in conflict"""
        return [var for var in self.variables
                if self.nconflicts(var, current[var], current) > 0]


#%% N-queens problem
def queen_constraint(A, a, B, b):
    """Constraint is satisfied (true) if A, B are really the same variable,
    or if they are not in the same row, down diagonal, or up diagonal."""
    return A == B or (a != b and A + a != B + b and A - a != B - b)

class NQueensCSP(CSP):
    """
    Make a CSP for the nQueens problem for search with min_conflicts.
    Suitable for large n, it uses only data structures of size O(n).
    Think of placing queens one per column, from left to right.
    That means position (x, y) represents (var, val) in the CSP.
    The main structures are three arrays to count queens that could conflict:
        rows[i]      Number of queens in the ith row (i.e. val == i)
        downs[i]     Number of queens in the \ diagonal
                     such that their (x, y) coordinates sum to i
        ups[i]       Number of queens in the / diagonal
                     such that their (x, y) coordinates have x-y+n-1 = i
    """

    def __init__(self, n):
        """Initialize data structures for n Queens."""
        CSP.__init__(self, list(range(n)), UniversalDict(list(range(n))),
                     UniversalDict(list(range(n))), queen_constraint)

        self.rows = [0] * n
        self.ups = [0] * (2 * n - 1)
        self.downs = [0] * (2 * n - 1)

    def nconflicts(self, var, val, assignment):
        """The number of conflicts, as recorded with each assignment.
        Count conflicts in row and in up, down diagonals. If there
        is a queen there, it can't conflict with itself, so subtract 3."""
        n = len(self.variables)
        c = self.rows[val] + self.downs[var + val] + self.ups[var - val + n - 1]
        if assignment.get(var, None) == val:
            c -= 3
        return c

    def assign(self, var, val, assignment):
        """Assign var, and keep track of conflicts."""
        old_val = assignment.get(var, None)
        if val != old_val:
            if old_val is not None:  # Remove old val if there was one
                self.record_conflict(assignment, var, old_val, -1)
            self.record_conflict(assignment, var, val, +1)
            CSP.assign(self, var, val, assignment)

    def unassign(self, var, assignment):
        """Remove var from assignment (if it is there) and track conflicts."""
        if var in assignment:
            self.record_conflict(assignment, var, assignment[var], -1)
        CSP.unassign(self, var, assignment)

    def record_conflict(self, assignment, var, val, delta):
        """Record conflicts caused by addition or deletion of a Queen."""
        n = len(self.variables)
        self.rows[val] += delta
        self.downs[var + val] += delta
        self.ups[var - val + n - 1] += delta


#%% Min-conflicts for CSPs
''' READ AND COMMENT to show your comprehensive understanding of the following function '''
def min_conflicts(csp, max_steps=100000):
    """See Figure 6.8 for the algorithm"""
    # Em đã chạy debug để hiểu hơn vể chức năng của các function và thuật toán
    csp.current = current = {}
    # Khơi tạo biến current để lưu lại các kết quả sau mỗi lần iterate rang(max_steps)
    for var in csp.variables:
        # Lấy biến (var) lần lượt từ 0 đến n-queens để thực hiện công việc tìm value
        # nào sẽ có xung đột thấp nhất
        val = min_conflicts_value(csp, var, current)
        # Sau khi tìm được min conflict value thì ta sẽ gán biến và giá trị vào current
        # (ví dụ {0: 7, 1: 1, 2: 3, 3: 5, 4: 0, 5: 2, 6: 4, 7: 6} value = {7, 1, 3, 5, 0, 2, 4, 6}
        #         1     0     0     0     0     1     0     0
        # Kết quả thu được khi debug iterate 25 8-queens
        csp.assign(var, val, current)
        # Gán var val
    for i in range(max_steps):
        # Lập cho đến khi có kết quả hoặc i == max_steps
        conflicted = csp.conflicted_vars(current)
        # Conflicted sẽ trải qua 2 function trung gian để kiểm tra biến trùng
        # nconflicted() và conflicted_vars()
        # Lấy vị trí các biến có conflicted như ví dụ trên thì ta có var = {0, 5}
        if not conflicted:
            # Khi có các vị trí ta kiểm tra xem, nếu current = rỗng thì kết thúc thuật toán -> solution
            return current
        var = random.choice(conflicted)
        # nếu current không rỗng thì ta sẽ random chọn biến
        val = min_conflicts_value(csp, var, current)
        # Sau đó ta tiếp tục thực hiện tìm giá trị conflicts nhỏ nhất và gán vào current
        # Thuật toán sẽ dừng khi tìm ra solution hoặc i == max_steps
        csp.assign(var, val, current)
    return None

def min_conflicts_value(csp, var, current):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("8-queens")
        self.setGeometry(10, 30, 1350, 1000)
        self.n_queens = 0

    def show_board(self, solution, nQ):
        self.n_queens = nQ
        label = QLabel("Wellcome guy", self)
        label.setGeometry(1100, 20, 200, 50)
        label.setFont(QFont('Roboto', 12))
        button_play = QPushButton("Play game", self)
        button_play.setGeometry(1100, 80, 200, 50)
        #button_play.clicked.connect(self.game_play)
        button_play = QPushButton("Reset", self)
        button_play.setGeometry(1100, 132, 200, 50)
        #button_play.clicked.connect(self.reset)
        y_axis = -10
        for x in range(self.n_queens):
            x_axis = 20
            y_axis += 12

            for y in range(self.n_queens):

                button = QPushButton("", self)
                button.setGeometry(x_axis, y_axis, 13, 13)
                x_axis += 12
                if solution[x][y] == -1:
                    button.setStyleSheet(u"background-color: rgb(52, 152, 219)")
                    #button.setIcon(QIcon("NguyenNgocThien_19110148_Week8_img_queens.png"))
                else:
                    button.setStyleSheet(u"background-color: white")
        self.show()

def boards(domains, n_queens):
    de_board = []
    for i in domains:
        de_board.append(domains[i])
    x = []
    y = []
    for i in range(n_queens):
        x.append(0)
        for j in range(n_queens):
            y.append(x)
    board = np.array(y)
    chess_board = board[0:n_queens:1]
    # create black and white button
    chess_board[0::2, 1::2] = 1  # gán ô theo hàng ngang xen kẽ giữa các ô
    chess_board[1::2, 0::2] = 1  # gán ô theo hàng dọc xen kẽ giữa các ô
    for i in range(n_queens):
        # set position of queens
        chess_board[domains[i], i] = -1
    return chess_board
#%% main
if __name__ == '__main__':
    # Do em không biết tại sao khi load 100*100 thì màng hình thì nó tự động tắt
    # nên em recommend nên set 80 quân hậu là chạy GUI/UI ok nhất ạ
    problem = NQueensCSP(n=80)
    min_conflicts(problem, max_steps=100000)
    print(problem.current)
    check = boards(problem.current, 80)
    print(check)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show_board(check, 80)
    sys.exit(app.exec())
