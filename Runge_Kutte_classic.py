import numpy as np

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

def f_classic(x, y):
    y = np.array(y)
    return np.array([A * y[1], -B * y[0]])

def k1_classic(h, x, y):
    return h * f_classic(x, y)

def k2_classic(h, x, y):
    return h * (f_classic(x + (1 / 2) * h, y + (1 / 2) * k1_classic(h, x, y)))

def k3_classic(h, x, y):
    return h * (f_classic(x + (1 / 2) * h, y + (1 / 2) * k2_classic(h, x, y)))

def k4_classic(h, x, y):
    return h * f_classic(x + h, y + k3_classic(h, x, y))

def Runge_Kutte_classic(h, x, y, x_half, y_half):
    r = 1
    h_eps = h
    while r > eps:
        h = h_eps
        while x < xk:
            y_prev = y
            y = y + (1 / 6) * (k1_classic(h, x, y) + 2 * k2_classic(h, x, y) + 2 * k3_classic(h, x, y) + k4_classic(h, x, y))
            x = x + h

            y_half = y_half + (1 / 6) * (k1_classic(h / 2, x_half, y_half) + 2 * k2_classic(h / 2, x_half, y_half) + 2 * k3_classic(h / 2, x_half, y_half) + k4_classic(h / 2, x_half, y_half))
            x_half = x_half + h / 2
            y_half = y_half + (1 / 6) * (k1_classic(h / 2, x_half, y_half) + 2 * k2_classic(h / 2, x_half, y_half) + 2 * k3_classic(h / 2, x_half, y_half) + k4_classic(h / 2, x_half, y_half))
            x_half = x_half + h / 2

        h_eps = h/2 * (((2**2 - 1)* eps/ (np.linalg.norm(y_half - y)))**(1/2))
        r = (y_half - y) / (2**(2) - 1)
        r = np.linalg.norm(r)

    return y_half

def Runge_Kutte_classic_auto(h, x, y, x_half, y_half):
    h_start = h
    r = 1
    local_error_eps = 10 ** (-5)
    while (x < xk):
        y_prev = y
        y_prev_half = y_half

        y = y + (1 / 6) * (k1_classic(h, x, y) + 2 * k2_classic(h, x, y) + 2 * k3_classic(h, x, y) + k4_classic(h, x, y))
        y_half = y_half + (1 / 6) * (k1_classic(h / 2, x_half, y_half) + 2 * k2_classic(h / 2, x_half, y_half) + 2 * k3_classic(h / 2, x_half, y_half) + k4_classic(h / 2, x_half, y_half))
        y_half = y_half + (1 / 6) * (k1_classic(h / 2, x_half, y_half) + 2 * k2_classic(h / 2, x_half, y_half) + 2 * k3_classic(h / 2, x_half, y_half) + k4_classic(h / 2, x_half, y_half))

        local_error = (y_half - y) / (1 - 2 ** (-4))
        if np.linalg.norm(local_error) > local_error_eps * (2 ** (4)):
            h = h / 2
            y = y_prev
            y_half = y_prev_half
        elif np.linalg.norm(local_error) > local_error_eps:
            y = y_half
            x = x + h
            h = h / 2
        elif np.linalg.norm(local_error) >= local_error_eps / (2 ** (5)):
            x = x + h
        else:
            x = x + h
            h = 2 * h

    return y_half

def Runge_Kutte_classic_auto_plots(h, x, y, x_half, y_half, x_plot, h_plot, local_error_classic):
    h_start = h
    r = 1
    local_error_eps = 10 ** (-5)
    while (x < xk):
        y_prev = y
        y_prev_half = y_half

        y = y + (1 / 6) * (k1_classic(h, x, y) + 2 * k2_classic(h, x, y) + 2 * k3_classic(h, x, y) + k4_classic(h, x, y))
        y_half = y_half + (1 / 6) * (k1_classic(h / 2, x_half, y_half) + 2 * k2_classic(h / 2, x_half, y_half) + 2 * k3_classic(h / 2, x_half, y_half) + k4_classic(h / 2, x_half, y_half))
        y_half = y_half + (1 / 6) * (k1_classic(h / 2, x_half, y_half) + 2 * k2_classic(h / 2, x_half, y_half) + 2 * k3_classic(h / 2, x_half, y_half) + k4_classic(h / 2, x_half, y_half))

        local_error = (y_half - y) / (1 - 2 ** (-4))
        t_local_error = true_local_error(x + h) - y
        if np.linalg.norm(local_error) > local_error_eps * (2 ** (4)):
            h = h / 2
            h_plot[len(h_plot) - 1] = h
            y = y_prev
            y_half = y_prev_half
        elif np.linalg.norm(local_error) > local_error_eps:
            y = y_half
            x = x + h
            x_plot.append(x)
            local_error_classic.append(np.linalg.norm(t_local_error) / np.linalg.norm(local_error))
            h = h / 2
            h_plot.append(h)
        elif np.linalg.norm(local_error) >= local_error_eps / (2 ** (5)):
            x = x + h
            x_plot.append(x)
            local_error_classic.append(np.linalg.norm(t_local_error) / np.linalg.norm(local_error))
            h_plot.append(h)
        else:
            x = x + h
            h = 2 * h
            x_plot.append(x)
            local_error_classic.append(np.linalg.norm(t_local_error) / np.linalg.norm(local_error))
            h_plot.append(h)

    return y_half, x_plot, h_plot, local_error_classic

def Runge_Kutte_classic_auto_plot3(h, x, y, x_half, y_half, x_plot, n):
    h_start = h
    r = 1
    local_error_eps = 10 ** (-n)
    while (x < xk):
        y_prev = y
        y_prev_half = y_half

        y = y + (1 / 6) * (k1_classic(h, x, y) + 2 * k2_classic(h, x, y) + 2 * k3_classic(h, x, y) + k4_classic(h, x, y))
        y_half = y_half + (1 / 6) * (k1_classic(h / 2, x_half, y_half) + 2 * k2_classic(h / 2, x_half, y_half) + 2 * k3_classic(h / 2, x_half, y_half) + k4_classic(h / 2, x_half, y_half))
        y_half = y_half + (1 / 6) * (k1_classic(h / 2, x_half, y_half) + 2 * k2_classic(h / 2, x_half, y_half) + 2 * k3_classic(h / 2, x_half, y_half) + k4_classic(h / 2, x_half, y_half))

        local_error = (y_half - y) / (1 - 2 ** (-4))
        if np.linalg.norm(local_error) > local_error_eps * (2 ** (4)):
            h = h / 2
            y = y_prev
            y_half = y_prev_half
        elif np.linalg.norm(local_error) > local_error_eps:
            y = y_half
            x = x + h
            x_plot.append(x)
            h = h / 2
        elif np.linalg.norm(local_error) >= local_error_eps / (2 ** (5)):
            x = x + h
            x_plot.append(x)
        else:
            x = x + h
            h = 2 * h
            x_plot.append(x)

    return y_half, len(x_plot)