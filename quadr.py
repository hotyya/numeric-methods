import numpy as np
import auxiliary_functions as af

n = 3
a = 1
b = 3
betha = 1/6

def left_rectangle(f, a, b, n, true_value):
    now_sum = 0
    h = (b - a) / n
    for i in range(1, n + 1):
        x = a + (i - 1) * h
        now_sum += f(x)
    now_sum = now_sum * h
    #if (abs(now_sum - true_value) < 0.01):
    #    print("integral: ", now_sum)
    #else:
    #    print("integral: ", now_sum)
    #    print("error: ", abs(true_value - now_sum))
    return abs(now_sum - true_value)

def mid_rectangle(f, a, b, n, true_value):
    now_sum = 0
    h = (b - a) / n
    for i in range(1, n + 1):
        x = a + (i - 0.5) * h
        now_sum += f(x)
    now_sum = now_sum * h
    #if (abs(now_sum - true_value) < 0.01):
    #    print("integral: ", now_sum)
    #else:
    #    print("integral: ", now_sum)
    #    print("error: ", abs(true_value - now_sum))
    return abs(now_sum - true_value)

def trapz(f, a, b, n, true_value):
    now_sum = 0
    h = (b - a) / n
    for i in range(1, n + 1):
        x1 = a + (i - 1) * h
        x2 = a + i * h
        now_sum += f(x1)
        now_sum += f(x2)
    now_sum = now_sum * (h / 2)
    
    return abs(now_sum - true_value)

#def simpson(f, a, b, n, true_value):
#    now_sum = 0
#    h = (b - a) / n
#    for i in range(1, n + 1):
#        x1 = a + (i - 1) * h
#        x2 = a + (i - 0.5) * h
#        x3 = a + i * h
#        now_sum += f(x1)
#        now_sum += f(x2)
#        now_sum += f(x3)
#    now_sum = now_sum * (h / 6)
#    if (abs(now_sum - true_value) < 0.01):
#        print("integral: ", now_sum)
#    else:
#        print("integral: ", now_sum)
#        print("error: ", abs(true_value - now_sum))

def simpson_(f, a, b, n, true_value):
    now_sum = 0
    x = np.linspace(a, b, n)
    h = (b - a) / n
    for i in range(1, len(x)):
        now_sum += (f(x[i]) + f(x[i - 1]) + 4 * f((x[i] + x[i - 1]) / 2)) * (h / 6)

    #print("integral: ", now_sum)
    return abs(now_sum - true_value)

def KF_gauss(func, a_segment, b_segment):
    I = 0

    m = np.zeros(2 * n)
    x_roots = np.zeros(n)

    m[0] = -6/5 * ((3 - b_segment) ** (5/6) - (3 - a_segment) ** (5/6))
    m[1] = ((3 - a_segment) ** (5/6) * (30 * a_segment + 108)) / 55 - ((3 - b_segment) ** (5/6) * (30 * b_segment + 108)) / 55
    m[2] = ((3 - a_segment) ** (5/6) * (330 * (a_segment) ** 2 + 1080 * a_segment + 3888)) / 935 - ((3 - b_segment) ** (5/6) * (330 * (b_segment) ** 2 + 1080 * b_segment + 3888)) / 935
    m[3] = ((3 - a_segment) ** (5/6) * (5610 * (a_segment) ** 3 + 17820 * (a_segment) ** 2 + 58320 * a_segment + 209952)) / 21505 - ((3 - b_segment) ** (5/6) * (5610 * (b_segment) ** 3 + 17820 * (b_segment) ** 2 + 58320 * b_segment + 209952)) / 21505
    m[4] = ((3 - a_segment) ** (5/6) * (129030 * (a_segment) ** 4 + 403920 * (a_segment) ** 3 + 1283040 * (a_segment) ** 2 + 4199040 * a_segment + 15116544)) / 623645 - ((3 - b_segment) ** (5/6) * (129030 * (b_segment) ** 4 + 403920 * (b_segment) ** 3 + 1283040 * (b_segment) ** 2 + 4199040 * b_segment + 15116544)) / 623645
    m[5] = ((3 - a_segment) ** (5/6) * (748374 * (a_segment) ** 5 + 2322540 * (a_segment) ** 4 + 7270560 * (a_segment) ** 3 + 23094720 * (a_segment) ** 2 + 75582720 * a_segment + 272097792)) / 4365515 - ((3 - b_segment) ** (5/6) * (748374 * (b_segment) ** 5 + 2322540 * (b_segment) ** 4 + 7270560 * (b_segment) ** 3 + 23094720 * (b_segment) ** 2 + 75582720 * b_segment + 272097792)) / 4365515

    am_sys = np.zeros((3, 3))
    b_sys = np.zeros(3)

    for i in range(n):
        for j in range(n):
            am_sys[i][j] = m[i + j]
        b_sys[i] = -m[n + i]
        
    a_sys = np.linalg.solve(am_sys, b_sys)
    a_sys = np.flip(a_sys)

    x_roots = af.cardano_formulas(a_sys)

    x_sys = np.zeros((3, 3))
    m_sys = np.zeros(3)

    for i in range(n):
        for j in range(n):
            x_sys[i][j] = x_roots[j]**i
        m_sys[i] = m[i]

    A_coeffs = np.linalg.solve(x_sys, m_sys)

    for i in range(n):
        I += A_coeffs[i] * func(x_roots[i])

    return I

def new_cotes(func, a_segment, b_segment):
    I = 0

    m = np.zeros(n)
    x = np.linspace(a_segment, b_segment, 3)

    m[0] = -6/5 * ((3 - b_segment) ** (5/6) - (3 - a_segment) ** (5/6))
    m[1] = ((3 - a_segment) ** (5/6) * (30 * a_segment + 108)) / 55 - ((3 - b_segment) ** (5/6) * (30 * b_segment + 108)) / 55
    m[2] = ((3 - a_segment) ** (5/6) * (330 * (a_segment) ** 2 + 1080 * a_segment + 3888)) / 935 - ((3 - b_segment) ** (5/6) * (330 * (b_segment) ** 2 + 1080 * b_segment + 3888)) / 935

    x_sys = np.zeros((3, 3))

    for i in range(n):
        for j in range(n):
            x_sys[i][j] = x[j] ** i

    A_coeffs = np.linalg.solve(x_sys, m)

    for i in range(n):
        I += A_coeffs[i] * func(x[i])

    return I