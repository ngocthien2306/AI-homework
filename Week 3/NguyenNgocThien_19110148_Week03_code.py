import numpy as np
import matplotlib.pyplot as py
import pandas
import random
import math

print("""Bai 1: Tạo một 2D numpy array và chạy thử các hàm max(), max(0), max(1). Cú pháp: ten_array.max(). 
Hãy cho biết công dụng khác biệt của các hàm này. """)
arr2d = np.array([[2, 5, 8], [1, 3, 7], [0, 0, -2], [10, 1, 0]])
print(arr2d)
print("In ra màng hình phần tử lớn nhất trong mảng! max =",arr2d.max())
print("In ra các phần tử lớn nhất của mỗi cột!",arr2d.max(0))
print("In ra các phần tử lớn nhất của mỗi hàng!",arr2d.max(1))

print("In ra phần tử nhỏ nhất trong mảng! min = ", arr2d.min())
print("In ra các phần tử nhỏ nhất của mỗi cột!", arr2d.min(0))
print("In ra các phần tử nhỏ nhất của mỗi hàng!",arr2d.min(1))
print("\n\n\n\n\n")

print("Bai 2: Vẽ đồ thị 3D cho hàm số z(x, y) = x2 + y2. Tùy ý vẽ một đồ thị hàm số z(x, y) khác. ")

def Equation(x, y):
    return x**2 + y**2
ax = py.figure()
x = y = np.arange(-10.0, 10.0, 0.1)
#X, Y = np.meshgrid()
#ax.plot_surface(X, Y)
#py.show()
print("\n\n\n\n\n")


print("Bai 3: Tìm hiểu package pandas và đọc dữ liệu từ 1 file dữ liệu dạng csv")

data = pandas.read_csv('pokemon_data.csv')

print(data)
print(data.info()) # lấy thông tin các cột (name column, kiểu dữ liệu)
print(data.head(5)) # lấy dữ liệu của n hàng đầu
head = data.head(5)
print(head.describe()) # thống kê dữ liệu gồm giá trị trung bình, max, min, số hàng


print(data.iloc[4: 6]) # lấy dữ liệu từ hàng n (4) đến hàng (6)
print(data.iloc[:: 100])
print(data.iloc[1, 2]) # lấy phần tử hàng 1 cột 2
print(data.iloc[[4]]) # lấy dữ liệu của hàng thứ n (2)


# tham khảo documentation
attack = np.array(data['Attack'])
py.plot(attack[1:10], '-go', label='HP') 
py.xlabel('X')
py.show();


print("\n\n\n\n\n")


print("Bai 4: Tìm hiểu về lập trình hướng đối tượng trong Python: cách tạo class, hàm constructor, thừa kế.")

class Course:
    def __init__(self, cid, cname, score):
        self.cid = cid
        self.cname = cname
        self.score = score
    def InputCourse(self):
        self.cid = input("Enter code of course: ")
        self.cname = input("Enter name of course: ")
        self.score = int(input("Enter score: "))
        print("\n\n\n\n")
    def OutputCourse(self):
        print("ID course:", self.cid)
        print("Name course:", self.cname)
        print("Score: ", self.score)
# Tham khảo cách kế thừa https://quantrimang.com/ke-thua-inheritance-trong-python-160258
class Student(Course):
    def __init__(self, sid, name, birthday):
        Course.InputCourse(self)
        Course.__init__(self, self.cid, self.cname, self.score)
        self.sid = sid
        self.name = name
        self.birthday = birthday
    def Info(self):
        print('Student id ', self.sid)
        print('Name ', self.name)
        print("Birthday", self.birthday)
    def Score(self):
        print("Report: ")
        self.Info()
        print('Course', self.cid, " - ", self.cname)
        print('Score', self.score)
        print("Result: ")
        if self.score < 5:
            print('Oh no no no no...You failed!')
        else:
            print("Oke! Congrats you passed!")

#course = Course("OOP1221AB", "OOP", 6)
student = Student(1, "Thien", "23/06/2001")
student.Score()