import Runge_Kutte_classic as RKc
import numpy as np
import matplotlib.pyplot as plt

y1_true = 0.317759
y2_true = -0.00973565

c = 1/17
A = 1/35
B = 1/10

eps = 10 ** (-4)
xk = np.pi

def true_local_error(x):
    y = np.zeros(2)
    y[0] = (np.sqrt(14) * np.pi * np.sin(x / (5 * np.sqrt(14)))) / 245 + (np.pi * np.cos(x / (5 * np.sqrt(14)))) / 10
    y[1] = (np.pi * np.cos(x / (5 * np.sqrt(14)))) / 35 - (7 * np.pi * np.sin(x / (5 * np.sqrt(14)))) / (10 * np.sqrt(14))

    return y

def f(x, y):
    y = np.array(y)
    return np.array([A * y[1], -B * y[0]])

def k1(h, x, y):
    return h * f(x, y)

def k2(h, x, y):
    return h * (f(x + c * h, y + c * k1(h, x, y)))

def twostep_Runge_Kutte_method(h, x, y, x_half, y_half, abs_error, x_plot_abs):
    r = 1
    h_eps = h
    while r > eps:
        h = h_eps
        while (x < xk):
            if x + h > xk:
                h = xk - x

            y_prev = y
            y = y + b[0] * k1(h, x, y) + b[1] * k2(h, x, y)
            x = x + h
            x_plot_abs.append(x)

            y_half = y_half + b[0] * k1(h/2, x_half, y_half) + b[1] * k2(h/2, x_half, y_half)
            x_half = x_half + h/2
            y_half = y_half + b[0] * k1(h/2, x_half, y_half) + b[1] * k2(h/2, x_half, y_half)
            x_half = x_half + h/2

            t_abs_error = true_local_error(x) - y_half
            abs_error.append(np.linalg.norm(t_abs_error))

        #print("y_half = ", y_half)
        h_eps = h/2 * (((2**2 - 1)* eps/ (np.linalg.norm(y_half - y)))**(1/2))
        #print("h_eps = ", h_eps)
        r = (y_half - y) / (2**(2) - 1)
        r = np.linalg.norm(r)
        print("|r| = ", r)

    return y_half, abs_error, x_plot_abs

def twostep_Runge_Kutte_automethod(h, x, y, x_half, y_half):
    h_start = h
    r = 1
    local_error_eps = 10 ** (-5)
    while (x < xk):
        y_prev = y
        y_prev_half = y_half

        y = y + b[0] * k1(h, x, y) + b[1] * k2(h, x, y)
        y_half = y_half + b[0] * k1(h/2, x_half, y_half) + b[1] * k2(h/2, x_half, y_half)
        y_half = y_half + b[0] * k1(h/2, x_half, y_half) + b[1] * k2(h/2, x_half, y_half)

        local_error = (y_half - y) / (1 - 2 ** (-2))
        if np.linalg.norm(local_error) > local_error_eps * 2 ** (2):
            h = h / 2
            y = y_prev
            y_half = y_prev_half
        elif np.linalg.norm(local_error) > local_error_eps:
            y = y_half
            x = x + h
            h = h / 2
        elif np.linalg.norm(local_error) >= local_error_eps / (2 ** (3)):
            x = x + h
        else:
            x = x + h
            h = 2 * h

    return y_half

def twostep_Runge_Kutte_automethod_plots(h, x, y, x_half, y_half, x_plot, h_plot, local_error_twostep):
    h_start = h
    r = 1
    local_error_eps = 10 ** (-5)
    while (x < xk):
        y_prev = y
        y_prev_half = y_half

        y = y + b[0] * k1(h, x, y) + b[1] * k2(h, x, y)
        y_half = y_half + b[0] * k1(h/2, x_half, y_half) + b[1] * k2(h/2, x_half, y_half)
        y_half = y_half + b[0] * k1(h/2, x_half, y_half) + b[1] * k2(h/2, x_half, y_half)

        t_local_error = true_local_error(x + h) - y
        local_error = (y_half - y) / (1 - 2 ** (-2))
        if np.linalg.norm(local_error) > local_error_eps * 2 ** (2):
            h = h / 2
            h_plot[len(h_plot) - 1] = h
            y = y_prev
            y_half = y_prev_half
        elif np.linalg.norm(local_error) > local_error_eps:
            y = y_half
            x = x + h
            x_plot.append(x)
            local_error_twostep.append(np.linalg.norm(t_local_error) / np.linalg.norm(local_error))
            h = h / 2
            h_plot.append(h)
        elif np.linalg.norm(local_error) >= local_error_eps / (2 ** (3)):
            x = x + h
            x_plot.append(x)
            local_error_twostep.append(np.linalg.norm(t_local_error) / np.linalg.norm(local_error))
            h_plot.append(h)
        else:
            x = x + h
            h = 2 * h
            x_plot.append(x)
            local_error_twostep.append(np.linalg.norm(t_local_error) / np.linalg.norm(local_error))
            h_plot.append(h)

    return y_half, x_plot, h_plot, local_error_twostep

def twostep_Runge_Kutte_automethod_plot3(h, x, y, x_half, y_half, x_plot, n):
    h_start = h
    r = 1
    local_error_eps = 10 ** (-n)
    while (x < xk):
        y_prev = y
        y_prev_half = y_half

        y = y + b[0] * k1(h, x, y) + b[1] * k2(h, x, y)
        y_half = y_half + b[0] * k1(h/2, x_half, y_half) + b[1] * k2(h/2, x_half, y_half)
        y_half = y_half + b[0] * k1(h/2, x_half, y_half) + b[1] * k2(h/2, x_half, y_half)

        local_error = (y_half - y) / (1 - 2 ** (-2))
        if np.linalg.norm(local_error) > local_error_eps * 2 ** (2):
            h = h / 2
            y = y_prev
            y_half = y_prev_half
        elif np.linalg.norm(local_error) > local_error_eps:
            y = y_half
            x = x + h
            x_plot.append(x)
            h = h / 2
        elif np.linalg.norm(local_error) >= local_error_eps / (2 ** (3)):
            x = x + h
            x_plot.append(x)
        else:
            x = x + h
            h = 2 * h
            x_plot.append(x)

    return y_half, len(x_plot)

x = 0
x_half = 0

y = np.array([B * np.pi, A * np.pi])
y_half = y

b = np.array([1 - 1 / (2 * c), 1 / (2 * c)])

h = (eps / ((1/(max(abs(x), abs(xk))))**(3) + np.linalg.norm(f(x, y))**(3))) ** (1/3)
h_half = h / 2

#print(twostep_Runge_Kutte_method(h, x, y, x_half, y_half))
print(twostep_Runge_Kutte_automethod(h, x, y, x_half, y_half))
print("______________________")
h = (eps / ((1/(max(abs(x), abs(xk))))**(5) + np.linalg.norm(f(x, y))**(5))) ** (1/5)
print(RKc.Runge_Kutte_classic(h, x, y, x_half, y_half))
print(RKc.Runge_Kutte_classic_auto(h, x, y, x_half, y_half))

h = (eps / ((1/(max(abs(x), abs(xk))))**(3) + np.linalg.norm(f(x, y))**(3))) ** (1/3)
x_plot = list()
x_plot.append(0)
h_plot = list()
h_plot.append(h)
local_error_twostep = list()
local_error_twostep.append(1)
y_, x_plot, h_plot, local_error_twostep = twostep_Runge_Kutte_automethod_plots(h, x, y, x_half, y_half, x_plot, h_plot, local_error_twostep)
print(y_)

h_classic = (eps / ((1/(max(abs(x), abs(xk))))**(5) + np.linalg.norm(f(x, y))**(5))) ** (1/5)
x_plot_classic = list()
x_plot_classic.append(0)
h_plot_classic = list()
h_plot_classic.append(h_classic)
local_error_classic = list()
local_error_classic.append(1)
y_classic, x_plot_classic, h_plot_classic, local_error_classic = RKc.Runge_Kutte_classic_auto_plots(h_classic, x, y, x_half, y_half, x_plot_classic, h_plot_classic, local_error_classic)
print(y_classic)

x_amount = list()
eps_order = list()
for i in range(1, 10):
    eps_order.append(i)
    x_plot3 = list()
    x_plot3.append(0)
    y_3, amount_x = twostep_Runge_Kutte_automethod_plot3(h, x, y, x_half, y_half, x_plot3, i)
    x_amount.append(amount_x)

x_amount_classic = list()
for i in range(1, 10):
    x_plot3 = list()
    x_plot3.append(0)
    y_3_classic, amount_x = RKc.Runge_Kutte_classic_auto_plot3(h, x, y, x_half, y_half, x_plot3, i)
    x_amount_classic.append(amount_x)

x_plot_abs = list()
x_plot_abs.append(0)
abs_error = list()
abs_error.append(0)
y_half, abs_error, x_plot_abs = twostep_Runge_Kutte_method(h, x, y, x_half, y_half, abs_error, x_plot_abs)

print()
print(local_error_twostep)
print(local_error_classic)
plt.figure()
plt.plot(x_plot, h_plot)
plt.plot(x_plot_classic, h_plot_classic)
plt.title("Зависимость величины шага интегрирования от x")

plt.figure()
plt.plot(x_plot, local_error_twostep)
plt.plot(x_plot_classic, local_error_classic)
plt.title("Зависимость отношения локальных погрешностей от x")

plt.figure()
plt.plot(eps_order, x_amount)
plt.plot(eps_order, x_amount_classic)
plt.title("Зависимость количества вычислений от заданной точности")

plt.figure()
plt.plot(x_plot_abs, abs_error)
plt.title("Зависимость истинной полной погрешности от от x")

plt.show()