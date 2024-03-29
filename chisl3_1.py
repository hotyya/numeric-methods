import numpy as np
import matplotlib.pyplot as plt
import quadr

true_chel = -41.88816344

n = 3
a = 1
b = 3
betha = 1/6

def func(x):
    return 3*np.cos(3.5*x)*np.exp((4*x)/3) + 2*np.sin(3.5*x)*np.exp(-(2*x)/3) + 4*x

def f(x):
  return 3 * np.cos(2.5 * x)*np.exp(7*x/4)+ 5* np.sin(0.5*x)* np.exp(3 * x/8) + 4

def F(x):
    return (3*np.cos(3.5*x)*np.exp(4*x/3) + 2*np.sin(3.5*x)*np.exp(-2*x/3) + 4*x) / ((b - x) ** betha)
# alpha = 0 betha = 1/6 a = 1 b = 3

true_value = -25.5814
y = list()
x = list()
for i in range(10, 100, 5):
    x.append(i)
    y.append(quadr.left_rectangle(func, 1, 3, i, true_value))

plt.figure()
plt.plot(x, y)
plt.title("Составная квадратурная формула левого прямоугольника")

y = list()
for i in range(10, 100, 5):
    y.append(quadr.mid_rectangle(func, 1, 3, i, true_value))

plt.figure()
plt.plot(x, y)
plt.title("Составная квадратурная формула среднего прямоугольника")

y = list()
for i in range(10, 100, 5):
    y.append(quadr.trapz(func, 1, 3, i, true_value))

plt.figure()
plt.plot(x, y)
plt.title("Составная квадратурная формула трапеции")

y = list()
for i in range(10, 100, 5):
    y.append(quadr.simpson_(func, 1, 3, i, true_value))

plt.figure()
plt.plot(x, y)
plt.title("Составная квадратурная формула Симпсона")

y = list()
x = list()
for i in range(10, 100, 5):
    x_ = np.linspace(a, b, i)
    x.append(i)
    I_gauss = 0
    for j in range(len(x_) - 1):
        I_gauss += quadr.KF_gauss(func, x_[j], x_[j + 1])
    y.append(abs(I_gauss - true_chel))

plt.figure()
plt.plot(x, y)
plt.title("Составная квадратурная формула Гаусса")

y = list()
x = list()
for i in range(10, 100, 5):
    x_ = np.linspace(a, b, i)
    x.append(i)
    I_new_cotes = 0
    for j in range(len(x_) - 1):
        I_new_cotes += quadr.new_cotes(func, x_[j], x_[j + 1])
    y.append(abs(I_new_cotes - true_chel))

plt.figure()
plt.plot(x, y)
plt.title("Составная квадратурная формула Ньютона-Котеса")


print("__________Метод Ричардсона__________")
x = np.linspace(a, b, 2)
h1 = x[1] - x[0]
I_gauss_1 = 0
for i in range(len(x) - 1):
    I_gauss_1 += quadr.new_cotes(func, x[i], x[i + 1])

x = np.linspace(a, b, 4)
h2 = x[1] - x[0]
I_gauss_2 = 0
for i in range(len(x) - 1):
    I_gauss_2 += quadr.new_cotes(func, x[i], x[i + 1])

x = np.linspace(a, b, 8)
h3 = x[1] - x[0]
I_gauss_3 = 0
for i in range(len(x) - 1):
    I_gauss_3 += quadr.new_cotes(func, x[i], x[i + 1])

amount = 8
m = - (np.log(abs((I_gauss_3 - I_gauss_2) / (I_gauss_2 - I_gauss_1)))) / (np.log(2))
print(m)

S = [I_gauss_1, I_gauss_2, I_gauss_3]
h = [h1, h2, h3]

R = 1
eps = 10 ** (-6)
r = 0

while R > eps:
    r += 1
    if r == 1:
        a_sys = np.array([[h[0] ** m, -1], [h[1] ** m, -1]])
        b_sys = -np.array([S[0], S[1]])

        J = np.linalg.solve(a_sys, b_sys)

        R = abs(J[-1] - S[1])
    elif r == 2:
        a_sys = [[h[0] ** m, h[0] ** (m + 1), -1], [h[1] ** m, h[1] ** (m + 1), -1], [h[2] ** m, h[2] ** (m + 1), -1]]
        b_sys = [-S[0], -S[1], -S[2]]

        J = np.linalg.solve(a_sys, b_sys)

        R = abs(J[-1] - S[-1])
    else:
        x = np.linspace(a, b, amount * 2)
        amount = amount * 2
        h.append(x[1] - x[0])
        I_gauss = 0
        for i in range(len(x) - 1):
            I_gauss += quadr.new_cotes(func, x[i], x[i + 1])

        S.append(I_gauss)
        m = - (np.log(abs((S[-1] - S[-2]) / (S[-2] - S[-3])))) / np.log(2)
        print(m)

        a_sys = np.zeros((len(S), len(S)))
        b_sys = -1*np.array(S)
        
        for i in range(len(S)):
            for j in range(len(S) - 1):
                a_sys[i][j] = h[i] ** (m + j)
            a_sys[i][len(S) - 1] = -1
            
        J = np.linalg.solve(a_sys, b_sys)

        R = abs(J[-1] - S[-1])

print("Длина шага, при котором была достигнута необходимая точность (Ньютон-Котес): ", h[-1])

print()

x = np.linspace(a, b, 2)
h1 = x[1] - x[0]
I_gauss_1 = 0
for i in range(len(x) - 1):
    I_gauss_1 += quadr.KF_gauss(func, x[i], x[i + 1])

x = np.linspace(a, b, 4)
h2 = x[1] - x[0]
I_gauss_2 = 0
for i in range(len(x) - 1):
    I_gauss_2 += quadr.KF_gauss(func, x[i], x[i + 1])

x = np.linspace(a, b, 8)
h3 = x[1] - x[0]
I_gauss_3 = 0
for i in range(len(x) - 1):
    I_gauss_3 += quadr.KF_gauss(func, x[i], x[i + 1])

amount = 8
m = - (np.log(abs((I_gauss_3 - I_gauss_2) / (I_gauss_2 - I_gauss_1)))) / (np.log(2))
print(m)

S = [I_gauss_1, I_gauss_2, I_gauss_3]
h = [h1, h2, h3]

R = 1
eps = 10 ** (-9)
r = 0

while R > eps:
    r += 1
    if r == 1:
        a_sys = np.array([[h[0] ** m, -1], [h[1] ** m, -1]])
        b_sys = -np.array([S[0], S[1]])

        J = np.linalg.solve(a_sys, b_sys)

        R = abs(J[-1] - S[1])
    elif r == 2:
        a_sys = [[h[0] ** m, h[0] ** (m + 1), -1], [h[1] ** m, h[1] ** (m + 1), -1], [h[2] ** m, h[2] ** (m + 1), -1]]
        b_sys = [-S[0], -S[1], -S[2]]

        J = np.linalg.solve(a_sys, b_sys)

        R = abs(J[-1] - S[-1])
    else:
        x = np.linspace(a, b, amount * 2)
        amount = amount * 2
        h.append(x[1] - x[0])
        I_gauss = 0
        for i in range(len(x) - 1):
            I_gauss += quadr.KF_gauss(func, x[i], x[i + 1])

        S.append(I_gauss)
        m = - (np.log(abs((S[-1] - S[-2]) / (S[-2] - S[-3])))) / np.log(2)
        print(m)

        a_sys = np.zeros((len(S), len(S)))
        b_sys = -1*np.array(S)
        
        for i in range(len(S)):
            for j in range(len(S) - 1):
                a_sys[i][j] = h[i] ** (m + j)
            a_sys[i][len(S) - 1] = -1
            
        J = np.linalg.solve(a_sys, b_sys)

        R = abs(J[-1] - S[-1])

print("Длина шага, при котором была достигнута необходимая точность (Гаусс): ", h[-1])

h_opt_rich = h[-1]

print()
print("____________Сетка с шагом h_opt___________")
print()

x = np.linspace(a, b, 2)
h1 = x[1] - x[0]
I_gauss_1 = 0
for i in range(len(x) - 1):
    I_gauss_1 += quadr.KF_gauss(func, x[i], x[i + 1])

x = np.linspace(a, b, 4)
h2 = x[1] - x[0]
I_gauss_2 = 0
for i in range(len(x) - 1):
    I_gauss_2 += quadr.KF_gauss(func, x[i], x[i + 1])

x = np.linspace(a, b, 8)
h3 = x[1] - x[0]
I_gauss_3 = 0
for i in range(len(x) - 1):
    I_gauss_3 += quadr.KF_gauss(func, x[i], x[i + 1])

m = - (np.log(abs((I_gauss_3 - I_gauss_2) / (I_gauss_2 - I_gauss_1)))) / (np.log(2))

a_sys = [[h[0] ** m, h[0] ** (m + 1), -1], [h[1] ** m, h[1] ** (m + 1), -1], [h[2] ** m, h[2] ** (m + 1), -1]]
b_sys = [-S[0], -S[1], -S[2]]

J = np.linalg.solve(a_sys, b_sys)

R = abs(J[-1] - S[-1])
eps = 10 ** (-6)

h_opt = h3 * ((eps / R) ** (1 / m))

x = np.linspace(a, b, int((b - a) / h_opt))
h1 = x[1] - x[0]
I_gauss_1 = 0
for i in range(len(x) - 1):
    I_gauss_1 += quadr.KF_gauss(func, x[i], x[i + 1])

x = np.linspace(a, b, 2 * int(((b - a) / h_opt)))
h2 = x[1] - x[0]
I_gauss_2 = 0
for i in range(len(x) - 1):
    I_gauss_2 += quadr.KF_gauss(func, x[i], x[i + 1])

x = np.linspace(a, b, 4 * int(((b - a) / h_opt)))
h3 = x[1] - x[0]
I_gauss_3 = 0
for i in range(len(x) - 1):
    I_gauss_3 += quadr.KF_gauss(func, x[i], x[i + 1])

amount = 4 * int(((b - a) / h_opt))
m = - (np.log(abs((I_gauss_3 - I_gauss_2) / (I_gauss_2 - I_gauss_1)))) / (np.log(2))

S = [I_gauss_1, I_gauss_2, I_gauss_3]
h = [h1, h2, h3]

R = 1
eps = 10 ** (-6)
r = 0

while R > eps:
    r += 1
    if r == 1:
        a_sys = np.array([[h[0] ** m, -1], [h[1] ** m, -1]])
        b_sys = -np.array([S[0], S[1]])

        J = np.linalg.solve(a_sys, b_sys)

        R = abs(J[-1] - S[1])
    elif r == 2:
        a_sys = [[h[0] ** m, h[0] ** (m + 1), -1], [h[1] ** m, h[1] ** (m + 1), -1], [h[2] ** m, h[2] ** (m + 1), -1]]
        b_sys = [-S[0], -S[1], -S[2]]

        J = np.linalg.solve(a_sys, b_sys)

        R = abs(J[-1] - S[-1])
    else:
        x = np.linspace(a, b, amount * 2)
        amount = amount * 2
        h.append(x[1] - x[0])
        I_gauss = 0
        for i in range(len(x) - 1):
            I_gauss += quadr.KF_gauss(func, x[i], x[i + 1])

        S.append(I_gauss)
        m = - (np.log(abs((S[-1] - S[-2]) / (S[-2] - S[-3])))) / np.log(2)

        a_sys = np.zeros((len(S), len(S)))
        b_sys = -1*np.array(S)
        
        for i in range(len(S)):
            for j in range(len(S) - 1):
                a_sys[i][j] = h[i] ** (m + j)
            a_sys[i][len(S) - 1] = -1
            
        J = np.linalg.solve(a_sys, b_sys)

        R = abs(J[-1] - S[-1])

print("Оптимальный шаг из пункта 2.3: ", h[-1])
print("Оптимальный шаг из пункта 2.2: ", h_opt_rich)
plt.show()