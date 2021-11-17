# Solve N-queens problems using AND-OR search algorithm
'''
YOUR TASKS:
1. Read the given code to understand
2. Implement the and_or_graph_search() function
3. (Optinal) Add GUI, animation...
'''
# Source code của thầy
# Em tái sử dụng lại code GUI tuần 8 để tiết kiệm thời gian.
import os
import sys
from collections import deque
import numpy as np
import random as rd
import time
from timeit import default_timer as timer

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


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


class NQueensProblem:
    """The problem of placing N queens on an NxN board with none attacking each other.
    A state is represented as an N-element array, where a value of r in the c-th entry means there is a queen at column c,
    row r, and a value of -1 means that the c-th column has not been filled in yet. We fill in columns left to right.

    Sample code: iterative_deepening_search(NQueensProblem(8))
    Result: <Node (0, 4, 7, 5, 2, 6, 1, 3)>
    """

    def __init__(self, N, initial):
        # self.initial = initial
        self.N = N
        self.initial = initial
    def actions(self, state):
        """In the leftmost empty column, try all non-conflicting rows."""
        if state[-1] != -1:
            return []  # All columns filled; no successors
        else:
            ran = rd.randint(0, len(state) - 1)
            col = state.index(-1)

            # return [(col, row) for row in range(self.N)
            return [row for row in range(self.N)
                    if not self.conflicted(state, row, col)]

    def goal_test(self, state):
        """Check if all columns filled, no conflicts."""
        if state[-1] != -1:
            return False
        return not any(self.conflicted(state, state[col], col)
                       for col in range(len(state)))

    def result(self, state, row):
        """Place the next queen at the given row."""
        col = state.index(-1)
        new = list(state[:])
        new[col] = row
        #Bạn Ngô Minh Đạt lớp chiều thứ 6 hướng dẫn em các fix lỗi này, em chạy debug mà không tìm được lỗi
        #if state[-1] != = 1 TypeError: 'int' object is not subscriptable
        #Vì các states phải là 1 mảng nên ở đây fix lại như sau [tuple(new)]
        #Vì goal_test phụ thuộc vào result nên mới có lỗi đó
        return [tuple(new)]

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

''' IMPLEMENT THE FOLLOWING FUNCTION '''

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.no_of_queens = 15
        self.loop = 0
        self.count = 0
        self.setWindowTitle("8-queens")
        self.setGeometry(100, 100, 1200, 850)
        self.solution = []
        self.origin_solution = []
        self.time = 0
        self.mess = ""
        self.init = ()
        #self.label_loop = QLabel("", self)

    def draw_board(self, result1):
        # draw board N*N
        x = []
        y = []
        for i in range(self.no_of_queens):
            x.append(0)
            for j in range(self.no_of_queens):
                y.append(x)
        board = np.array(y)
        chess_board = board[0: self.no_of_queens:1]
        # create black and white button
        chess_board[0::2, 1::2] = 1  # gán ô theo hàng ngang xen kẽ giữa các ô
        chess_board[1::2, 0::2] = 1  # gán ô theo hàng dọc xen kẽ giữa các ô
        for i in range(self.no_of_queens):
            # set position of queens
            chess_board[i, result1[i]] = -1

        print(chess_board)

        return chess_board

    def log(self):

        label_so = QLabel("Time: " + str(self.time) +"s", self)
        label_so.setGeometry(850, 250, 170, 40)
        label_so.setFont(QFont('Roboto', 12))

        label_so1 = QLabel(str(self.mess), self)
        label_so1.setGeometry(850, 290, 220, 60)
        label_so1.setFont(QFont('Roboto', 10))

        label_so2 = QLabel("Initial position: " + str(self.init), self)
        label_so2.setGeometry(850, 320, 220, 60)
        label_so2.setFont(QFont('Roboto', 10))

        mess = ""
        for i in self.solution:
           mess += str(i) + " -> "
        txt = QTextEdit(mess + "Result by column", self)
        txt.setGeometry(850, 385, 310, 110)
        txt.setFont(QFont('Roboto', 10))

        mess1 = ""
        for i in self.origin_solution:
           mess1 += str(i) + " -> "
        txt1 = QTextEdit(mess1 + "Result by row", self)
        txt1.setGeometry(850, 520, 300, 110)
        txt1.setFont(QFont('Roboto', 10))

    def game_play(self, result, time, mess, init):
        self.hide()
        self.time = time
        self.mess = mess
        self.origin_solution = result
        self.solution = result
        self.init = init
        self.show_board(result)
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

    def show_board(self, result):
        board = self.draw_board(result)
        y_axis = -30
        #Right side
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
            label = QLabel(str(x), self)
            label.setGeometry(2, y_axis, 51, 51)
            for y in range(self.no_of_queens):
                label = QLabel(str(y), self)
                label.setGeometry(x_axis + 18, -15, 51, 51)
                button = QPushButton("", self)
                button.setGeometry(x_axis, y_axis, 51, 51)
                x_axis += 50
                if board[x][y] == -1:
                    button.setStyleSheet(u"background-color: rgb(52, 152, 219)")
                    #button.setIcon(QIcon("NguyenNgocThien_19110148_Week8_img_queens.png"))
                    a = x, y
                    solution.append(a)
                elif board[x][y] == 1:
                    button.setStyleSheet(u"background-color: rgba(44, 62, 80, 0.5)")
                else:
                    button.setStyleSheet(u"background-color: white")
        self.solution = solution
        self.log()

def and_or_graph_search(problem, result1):
        """See [Figure 4.11] for the algorithm"""
        return or_search(problem.initial, problem, [], result1)

def or_search(state, problem, path, result1):
        if problem.goal_test(state):
            return [[], "Back"]
        if state in path:
            return "failure"
        for action in problem.actions(state):
            path.insert(0, state)
            plan = and_search(problem.result(state, action), problem, path, result1)
            if plan != "failure":
                plan.insert(0, action)
                return plan
        return "failure"

def and_search(states, problem, path, result1):
        solution = []
        for si in states:
            plan = or_search(si, problem, path, result1)
            if plan == "failure":
                return "failure"
            solution.append("if (" + "#" .join((str(item) for item in states)) + ") then " + str(plan))
        for item in states:
            result1.append(item)
        new_list = [s.replace("\\", "") for s in solution]

        print(new_list)
        return new_list
def solution(result, N):
    res = []
    for y in result[0]:
        res.append(y)
    for x in range(0, N):
        if x not in res:
            res[len(res) - 1] = x
    return res
def check_solution(res):
    if -1 in res:
        return "Find solution fail! please \ntry with other initial \nlocation"
    return "Find solution successful ^^"
if __name__ == '__main__':
    no_of_queens = 15
    #Set initial position
    #Để giải quyết vấn đề lúc nào plan cũng bắt đầu từ số 0 thì em
    #sẻ cho người sử dụng nhập tự nhập vào số hàng và cột muốn giải
    #nếu có thời gian em sẻ phát triển thêm cho người dùng nhập tự giao diện (GUI)
    #và xuất ra tất cả vị trị có thể tìm ra solution
    print(no_of_queens, "queens")
    #col = int(input("Enter the column you want to start: "))
    #row = int(input("Enter the row you want to start: "))
    ran_col = rd.randint(0, no_of_queens - 1)
    ran_row = rd.randint(0, no_of_queens - 1)
    initial = tuple([-1] * no_of_queens)  # -1: no queen in that column
    lst = list(initial)
    lst[ran_col] = ran_row
    initial = tuple(lst)
    #==================
    result1 = []
    #Measure time
    start = timer()
    problem = NQueensProblem(no_of_queens, initial)
    result = and_or_graph_search(problem, result1)
    end = timer()
    deltaT = end - start
    #Print and draw result
    print(result1[0])
    messeage = check_solution(result1)
    print(result)
    solution_final = solution(result1, no_of_queens)
    print(solution_final)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.game_play(solution_final, round(deltaT, 3), messeage, (ran_col, ran_row))
    sys.exit(app.exec())












