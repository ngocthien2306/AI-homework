
#===========================================================================================================================
#Bai tap 1: Viết hàm giải phương trình bậc 2 với các tham số của hàm là hệ số của phương trình. 
import math
def Equation(a,b,c): 
    try:
        if a == 0:
            if b == 0:
                print("The equation no solution")
            else:
                print("The equation has one solution: ", -c/b)
        else:
            delta = b*b - 4*a*c
            if delta > 0:
                x1 = (float)((b - math.sqrt(delta)) / 2*a)
                x2 = (float)((-b - math.sqrt(delta)) / 2*a)
                print("The equation have two solution x1 = ", x1, "and x2 = ", x2)
            elif delta == 0:
                print("The equation has one solution x1 = x2 = ", -b/2*a)
            else:
                print("The equation no sulution")
    except:
        print("Error!")
try:
    a = float(input("Enter number a: "))
    b = float(input("Enter number b: "))
    c = float(input("Enter number a: "))
except:
    print("Please import number!")
print("Exercise 1: giải phương trình bậc 2 ")
Equation(a, b, c)



#=============================================================================================================================
#Bai tap 2: Viết hàm tính giai thừa của một số nguyên dương.
#  Nếu người dùng nhập số âm thì yêu cầu nhập lại đến khi nhập số không âm. Yêu cầu: dùng vòng lặp while.
print("Exercise 2: Tính giai thừa của một số nguyên dương")
def Fac(n):
    if n == 0:
        return 1
    return n * Fac(n - 1)
number = float(input("Nhap 1 so nguyen duong: "))

while number < 0:
    number = int(input("Nhap 1 so nguyen duong: "))
if number > 0:
     print(number, " ! = ", Fac(number))

#=============================================================================================================================
#Bai tap 3: Viết hàm trả về mảng các phần tử chia hết cho 5 trong một mảng (lưu bằng list). 
import numpy
print("Exercise 3: Viết hàm trả về mảng các phần tử chia hết cho 5 trong một mảng")
element = []
def divideBy5(arr):
    try:
        for i in range(0, len(arr)):
            if int(arr[i]) % 5 == 0:
                element.append(arr[i])
        return element;
    except:
        return -1
arr = [-5, 5, 10, 12, 14, 4, 12.4]
if divideBy5(arr) == -1:
    print("Error! the element in the list wrong format")
else:
    print(element)

#=============================================================================================================================
#Bai tap 4: Cho numpy array có các phần tử: [-2 6 3 10 15 48], 
#viết lệnh (không dùng vòng lặp) để lấy ra các bộ phần tử: [3 15], [6 10 48], [10 15 48], [48 15 10].
import numpy as np
print("""Exercise 4: Cho numpy array có các phần tử: [-2 6 3 10 15 48], viết lệnh 
(không dùng vòng lặp) để lấy ra các bộ phần tử: [3 15], [6 10 48], [10 15 48], [48 15 10].\n""")
print("Solution")
array = np.array([-2, 6, 3, 10, 15, 48])
print(array)
print(array[2::2])
print(array[1::2])
print(array[3::])
print(array[5:2:-1])

#=============================================================================================================================
#Bai tap 5: Viết hàm có hai tham số là một mảng số bất kỳ và một tham số boolean tên SapXepTang. Nếu tham số 
# SapXepTang có giá trị True thì sắp xếp mảng tăng dần, có giá trị False thì sắp giảm dần. Yêu cầu: dùng
#  numpy array để chứa mảng.

print("Exercise 5: Viết hàm có hai tham số là một mảng số bất kỳ và một tham số boolean tên SapXepTang. Nếu tham số SapXepTang có giá trị True thì sắp xếp mảng tăng dần, có giá trị False thì sắp giảm dần. Yêu cầu: dùng numpy array để chứa mảng.")
import numpy
def Sort(array, check_sorting):
    if check_sorting == 1:
        print("The array is increasing")
        for i in range(0, len(array)):
            for j in range(0, len(array) - i - 1):
                if array[j] > array[j + 1]:
                    temp = array[j]
                    array[j] = array[j + 1]
                    array[j + 1] = temp
        return array
    else:
        print("The array is decreasing")
        for i in range(0, len(array)):
            for j in range(0, len(array) - i - 1):
                if array[j] < array[j + 1]:
                    temp = array[j]
                    array[j] = array[j + 1]
                    array[j + 1] = temp
        return array

data = numpy.array([1, 6, 12, 16, 3, 2, 43, 7, 9, 22, 11, 5 , 3, 99])
check_sorting = True;
print(Sort(data, check_sorting))
print(Sort(data, check_sorting = False))

print("The array is increasing: ",sorted(data, reverse=True))
print("The array is decreasing: ",sorted(data, reverse=False))
