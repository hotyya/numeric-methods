import numpy as np

def cardano_formulas(coeffs):
    y = np.zeros(3)
    x = np.zeros(3)

    a = 1
    b = coeffs[0]
    c = coeffs[1]
    d = coeffs[2]

    p = ((3 * a * c - b**2) / 3 * a**2) / 3
    q = ((2 * b**3) / (27 * a**3) - (b * c) / (3 * a**2) + d / a)

    if q < 0:
        r = -np.sqrt(abs(p))
    else:
        r = np.sqrt(abs(p))
    phi = np.arccos(q / (r**3))

    y[0] = -2*r*np.cos(phi / 3)
    y[1] = 2*r*np.cos(np.pi / 3 - phi / 3)
    y[2] = 2*r*np.cos(np.pi / 3 + phi / 3)

    x = y - b / (3 * a)

    return x
