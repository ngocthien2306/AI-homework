# The following code implements A* search to solve the path finding problem in a 10x10 maze.
# However, it has some BUGS leading to infinite loops and nonoptimal solutions!
'''
DEBUG the code to make it work with the maze map given in the exercise.
Hint: You might want to print the current_node and the closed_list (explored set) for each loop
      to check if the current_node is in the closed_list.
'''
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import os
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("A star")
        self.setGeometry(100, 100, 1024, 800)
        self.board = []
        self.solution = []
        self.init = []
        self.goal = []
        self.cell = []
        self.heuristic = []
        self.full_path = []
    """ 
    em đã tham khảo cách tạo button bằng code ở trang này https://www.geeksforgeeks.org/pyqt5-change-color-of-push-button/ 
    sau đó nãy ra ý tưởng dùng vòng lập for để tạo ra matran là các button
    """
    def checkBoard(self, board, solution, init, goal, cell, heuristic, full_path):
        self.board = board
        self.solution = solution
        self.init = init
        self.goal = goal
        self.cell = cell
        self.heuristic = heuristic
        self.full_path = full_path
        b = -40
        l = 20
        button_so = QPushButton("Solution", self)
        button_so.setGeometry(660, 100, 189, 39)
        button_so.clicked.connect(self.solutionPath)
        button_full = QPushButton("Full path", self)
        button_full.setGeometry(660, 140, 189, 39)
        button_full.clicked.connect(self.fullPath)
        button_new = QPushButton("Reset", self)
        button_new.setGeometry(660, 180, 189, 39)
        button_new.clicked.connect(self.reset)

        for x in range(0, len(board)):
            a = 20
            b += 60
            label_y = QLabel(str(x), self)
            label_y.setGeometry(4, b, 61, 61)
            label_y.setStyleSheet(u"background-color: lightblue")
            for y in range(0, len(board)):
                button = QPushButton("", self)
                button.setGeometry(a, b, 61, 61)
                button.setStyleSheet(u"background-color: white")
                a += 60
                if board[x][y] == 1:
                    button.setIcon(QIcon('NguyenNgocThien_19110148_wall.png'))
                    button.setStyleSheet(u"background-color: rgb(133, 146, 158)")
                if (x, y) == init:
                    button.setStyleSheet(u"background-color: rgb(241, 148, 138)")
                if (x, y) == goal:
                    button.setIcon(QIcon('NguyenNgocThien_19110148_queen.png'))
                    button.setStyleSheet(u"background-color: rgb(130, 188, 247)")
        for z in range(len(board)):
            label_x = QLabel(str(z), self)
            label_x.setGeometry(l, -10, 61, 35)
            label_x.setStyleSheet(u"background-color: lightblue")
            l += 60

        label = QLabel("Wellcome player 1", self)
        label.setFont(QFont('Roboto', 20))
        label.setGeometry(660, 20, 280, 40)
        self.log()
        self.show()
    def reset(self):
        self.hide()
        b = -40
        for x in range(0, len(self.board)):
            a = 20;
            b += 60
            for y in range(0, len(self.board)):
                button = QPushButton("", self)
                button.setGeometry(a, b, 61, 61)
                button.setStyleSheet(u"background-color: white")
                a += 60
                if self.board[x][y] == 1:
                    button.setIcon(QIcon('NguyenNgocThien_19110148_wall.png'))
                    button.setStyleSheet(u"background-color: rgb(133, 146, 158)")
                if (x, y) == self.init:
                    button.setIcon(QIcon('knight.png'))
                    button.setStyleSheet(u"background-color: rgb(241, 148, 138)")
                if (x, y) == self.goal:
                    button.setIcon(QIcon('NguyenNgocThien_19110148_queen.png'))
                    button.setStyleSheet(u"background-color: rgb(130, 188, 247)")
                if (x, y) in self.solution:
                    button.setStyleSheet(u"background-color: white")
        self.show()
    def solutionPath(self):
        self.hide()
        b = -40
        for x in range(0, len(self.board)):
            a = 20;
            b += 60
            for y in range(0, len(self.board)):
                button = QPushButton("", self)
                button.setGeometry(a, b, 61, 61)
                button.setStyleSheet(u"background-color: white")
                a += 60
                if self.board[x][y] == 1:
                    button.setIcon(QIcon('NguyenNgocThien_19110148_wall.png'))
                    button.setStyleSheet(u"background-color: rgb(133, 146, 158)")
                if (x, y) == self.init:
                    button.setStyleSheet(u"background-color: rgb(241, 148, 138)")
                if (x, y) == self.goal:
                    button.setIcon(QIcon('NguyenNgocThien_19110148_queen.png'))
                    button.setStyleSheet(u"background-color: rgb(130, 188, 247)")
                if (x, y) in self.solution:
                    button.setIcon(QIcon('NguyenNgocThien_19110148_step.png'))
                    button.setStyleSheet(u"background-color: rgb(250, 219, 216)")
        self.show()
    def fullPath(self):
        self.hide()
        b = -40
        for x in range(0, len(self.board)):
            a = 20
            b += 60
            for y in range(0, len(self.board)):
                button = QPushButton("", self)
                button.setGeometry(a, b, 61, 61)
                button.setStyleSheet(u"background-color: white")
                a += 60
                if self.board[x][y] == 1:
                    button.setIcon(QIcon('NguyenNgocThien_19110148_wall.png'))
                    button.setStyleSheet(u"background-color: rgb(133, 146, 158)")
                if (x, y) == self.init:
                    button.setStyleSheet(u"background-color: rgb(241, 148, 138)")
                if (x, y) == self.goal:
                    button.setIcon(QIcon('NguyenNgocThien_19110148_queen.png'))
                    button.setStyleSheet(u"background-color: rgb(130, 188, 247)")
                if (x, y) in self.full_path:
                    button.setIcon(QIcon('NguyenNgocThien_19110148_step.png'))
                    button.setStyleSheet(u"background-color: rgb(250, 219, 216)")
        self.show()
    def log(self):
        path_1 = ""
        full_path_1 = ""
        f = open("log.txt", "r")
        for i in self.solution:
            path_1 += str(i) + " -> "
        for i in self.full_path:
            full_path_1 += str(i) + " -> "
        label_mes = QLabel("Message - Exercise 1", self)
        label_mes.setFont(QFont('Roboto', 12))
        label_mes.setGeometry(660, 250, 200, 38)
        text_box = QTextEdit(f.read(), self)
        text_box.setGeometry(660, 300, 300, 200)
        label_so = QLabel("Solution", self)
        label_so.setFont(QFont('Roboto', 12))
        label_so.setGeometry(660, 640, 100, 38)
        text_so = QTextEdit(str(self.init) + " -> " + path_1 + str(self.goal), self)
        text_so.setGeometry(660, 680, 300, 80)
        label_pos = QLabel("Full path", self)
        label_pos.setFont(QFont('Roboto', 12))
        label_pos.setGeometry(20, 640, 100, 38)
        text_full_so = QTextEdit(full_path_1, self)
        text_full_so.setGeometry(20, 680, 600, 80)

class Node():
    """A node class for A* search"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0  # PATH-COST to the node
        self.h = 0  # heuristic to the goal: straight-line distance hueristic
        self.f = 0  # evaluation function f(n) = g(n) + h(n)

    def __eq__(self, other):
        return self.position == other.position


''' DEBUG THE FOLLOWING FUNCTION '''


def astar(maze, start, end, cell, heuristic, full_path):
    """Returns a list of tuples as a solution from "start" to "end" state in "maze" map using A* search.
    See lecture slide for the A* algorithm."""


    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []  # frontier queue
    closed_list = []  # explored set

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Check if we found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path

        # Expansion: Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (
                    len(maze[len(maze) - 1]) - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children

        for child in children:
            # Child is on the closed list
            # Tham khảo của bạn Ngô Minh Đạt - 19110115 Lớp thứ 6
            # Nhưng em có 1 chút chỉnh sửa, không cần break vòng lập thì vẫn chạy được, em đã test hết 2 trường hợp
            check_closed = False
            for closed_child in closed_list:
                if child == closed_child:
                    """ 
                    Vòng lập for sẻ lập vô tận tại vị trí (3, 4) vì tính chất của continue là bỏ qua một phần vòng khi một điều kiện được áp ứng,
                    không thể đi trở lại các vị trí trước vì thế ta dùng 1 biến check_closed để continue khi chạy hết một vòng lập. 
                    #print("child: ", child.position)
                    #print("closed_child: ", closed_child.position)
                    #continue 
                    """
                    #print tất cá các vị trí từng đi qua
                    #print(child.position)
                    full_path.append(child.position)
                    check_closed = True
            if check_closed:
                continue
            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h
            cell.append(child.position)
            heuristic.append(child.f)

            f = open("log.txt", "a")
            f.write(str(child.position) + " -> ")
            f.write(str(child.f) + ";\n")
            f.close()
            # Child is already in the open list
            check_open = False
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    check_open = True
            if check_open:
                continue
            # Add the child to the open list
            open_list.append(child)

if __name__ == '__main__':
    ''' CHANGE THE BELOW VARIABLE TO REFLECT TO THE MAZE MAP IN THE EXERCISE '''
    maze = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 1, 0, 0],  # 1: obstacle position
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0]]

    maze1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 1, 1, 1, 0],  # 1: obstacle position
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]

    os.remove("log.txt")
    start = (0, 0)
    goal = (8, 9)
    cell = []
    heuristic = []
    full_path = []
    path = astar(maze1, start, goal, cell, heuristic, full_path)
    #delete first item and last item
    del path[0]
    del path[len(path) - 1]

    # create pyqt5 app
    App = QApplication(sys.argv)
    # create the instance of our Window
    window = MainWindow()
    full_path.append(path[len(path) - 1])
    full_path.append(path[len(path) - 2])
    del full_path[0]
    window.checkBoard(maze1, path, start, goal, cell, heuristic, full_path)
    # start the app
    sys.exit(App.exec())