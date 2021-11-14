# Solve N-queens problems using Simulated annealing algorithm
'''
YOUR TASKS:
1. Read the code to understand
2. Implement the simulated_annealing() function
3. (Optinal) Try other shedule() functions
4. (Optinal) Add GUI, animation...
'''
#File code của thầy
import sys
from collections import deque
from PyQt5.QtGui import *
from  PyQt5.QtWidgets import *
import numpy as np
import random as rd
import math
class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        # next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        next_node = Node(next_state, self, action)
        return next_node

# ______________________________________________________________________________

class NQueensProblem:
    """The problem of placing N queens on an NxN board with none attacking each other.
    A state is represented as an N-element array, where a value of r in the c-th entry means there is a queen at column c,
    row r, and a value of -1 means that the c-th column has not been filled in yet. We fill in columns left to right.

    Sample code: iterative_deepening_search(NQueensProblem(8))
    Result: <Node (0, 4, 7, 5, 2, 6, 1, 3)>
    """

    def __init__(self, N):
        # self.initial = initial
        self.initial = tuple([-1] * N)  # -1: no queen in that column
        self.N = N

    def actions(self, state):
        """In the leftmost empty column, try all non-conflicting rows."""
        if state[-1] != -1:
            return []  # All columns filled; no successors
        else:
            col = state.index(-1)
            # return [(col, row) for row in range(self.N)
            return [row for row in range(self.N)
                    if not self.conflicted(state, row, col)]

    def result(self, state, row):
        """Place the next queen at the given row."""
        col = state.index(-1)
        new = list(state[:])
        new[col] = row
        return tuple(new)

    def conflicted(self, state, row, col):
        """Would placing a queen at (row, col) conflict with anything?"""
        return any(self.conflict(row, col, state[c], c)
                   for c in range(col))

    def conflict(self, row1, col1, row2, col2):
        """Would putting two queens in (row1, col1) and (row2, col2) conflict?"""
        return (row1 == row2 or  # same row
                col1 == col2 or  # same column
                row1 - col1 == row2 - col2 or  # same \ diagonal
                row1 + col1 == row2 + col2)  # same / diagonal

    def value(self, node):
        """Return (-) number of conflicting queens for a given node"""
        num_conflicts = 0
        for (r1, c1) in enumerate(node.state):
            for (r2, c2) in enumerate(node.state):
                if (r1, c1) != (r2, c2):
                    num_conflicts += self.conflict(r1, c1, r2, c2)

        return -num_conflicts


''' USE OTHER SCHEDULE FUNCTION IF YOU WANT TO '''


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.no_of_queens = 15
        self.loop = 0
        self.count = 0
        self.setWindowTitle("8-queens")
        self.setGeometry(100, 100, 1200, 850)
        self.solution = []
        self.label_loop = QLabel("", self)
        self.game_play()
    def draw_board(self):
        #draw board N*N
        x = []
        y = []
        problem1 = NQueensProblem(self.no_of_queens)
        result1 = self.simulated_annealing(problem1)
        print(result1.state)
        for i in range(self.no_of_queens):

            x.append(0)
            for j in range(self.no_of_queens):
                y.append(x)
        board = np.array(y)
        chess_board = board[0: self.no_of_queens:1]
        #create black and white button
        chess_board[0::2, 1::2] = 1 # gán ô theo hàng ngang xen kẽ giữa các ô
        chess_board[1::2, 0::2] = 1 # gán ô theo hàng dọc xen kẽ giữa các ô
        for i in range(self.no_of_queens):
            #set position of queens
            chess_board[result1.state[i], i] = -1

        print(chess_board)

        return chess_board
 
    def log(self):
        mess = ""
        for i in self.solution[1:]:
            mess += str(i) + " "
        label_so = QLabel("Solution", self)
        label_so.setGeometry(850, 250, 100, 60)
        label_so.setFont(QFont('Roboto', 12))
        self.label_loop = QLabel("Retry: " + str(self.loop) + " times " + str(self.count), self)
        self.label_loop.setGeometry(850, 480, 200, 60)
        self.label_loop.setFont(QFont('Roboto', 12))
        txt = QTextEdit(mess, self)
        txt.setGeometry(850, 330, 250, 100)
        self.loop = 0
        self.count = 0

    def game_play(self):
        self.hide()
        self.show_board()
        self.show()
    def reset(self):
        self.hide()
        y_axis = -30
        for x in range(self.no_of_queens):
            x_axis = 20
            y_axis += 50
            for y in range(self.no_of_queens):
                button = QPushButton("", self)
                button.setGeometry(x_axis, y_axis, 51, 51)
                button.setStyleSheet(u"background-color: white")

                x_axis += 50


        self.show()
    def show_board(self):
        board = self.draw_board()
        y_axis = -30
        label = QLabel("Wellcome guy", self)
        label.setGeometry(850, 20, 200, 50)
        label.setFont(QFont('Roboto', 12))
        button_play = QPushButton("Play game", self)
        button_play.setGeometry(850, 80, 200, 50)
        button_play.clicked.connect(self.game_play)
        button_play = QPushButton("Reset", self)
        button_play.setGeometry(850, 132, 200, 50)
        button_play.clicked.connect(self.reset)
        solution = [[]]
        for x in range(self.no_of_queens):
            x_axis = 20
            y_axis += 50
            for y in range(self.no_of_queens):
                button = QPushButton("", self)
                button.setGeometry(x_axis, y_axis, 51, 51)
                x_axis += 50
                if board[x][y] == -1:
                    button.setStyleSheet(u"background-color: rgb(52, 152, 219)")
                    button.setIcon(QIcon("NguyenNgocThien_19110148_Week8_img_queens.png"))
                    a = x, y
                    solution.append(a)
                elif board[x][y] == 1:
                    button.setStyleSheet(u"background-color: rgba(44, 62, 80, 0.5)")
                else:
                    button.setStyleSheet(u"background-color: white")
        self.solution = solution
        self.label_loop.deleteLater()
        self.log()

    def simulated_annealing(self, problem):
        """See [Figure 4.5] for the algorithm."""
        current = Node(problem.initial)
        for t in range(1, 1000000):
            T = schedule(t)
            if T == 0:
                return current
            successor = current.expand(problem)

            if len(successor) == 0:
                '''when lenght of successor = 0 and count state of current node value = -1 > 0 
                set current initial to run again, when has solution or stop condition then wen stop'''
                if current.state.count(-1) > 0:
                    current = Node(problem.initial)
                    self.loop += 1
            else:
                next = successor[rd.randint(0, len(successor) - 1)]
                deltaE = problem.value(next) - problem.value(current)
                if deltaE > 0:
                    self.count += 1
                    current = next
                else:
                    probability = math.e ** (deltaE / T)
                    if probability == 1:
                        current = next




def schedule(t, k=20, lam=0.002, limit=10000):
    """One possible schedule function for simulated annealing"""
    return (k * np.exp(-lam * t) if t < limit else 0)


''' WRITE THIS FUNCTION: '''




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
