import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy import interpolate

def gen_data(t, a, b, c, d, e, f, noise=0, n_outliers=0, random_state=0):
    y = f - a*np.log(t/b + 1.0) - c*t - d*np.exp(e*t)
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(t.size)
    outliers = rnd.randint(0, t.size, n_outliers)
    error[outliers] *= 10
    return y + error

def g_data(t, a, b, noise=0, n_outliers=0, random_state=0):
    y = a + b*t
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(t.size)
    outliers = rnd.randint(0, t.size, n_outliers)
    error[outliers] *= 10
    return y + error

def fun(x, t, y):
    return x[5] - x[0]*np.log(t/x[1] + 1.0) - x[2]*t - x[3]*np.exp(x[4]*t) - y

def funL(x, t, y):
    return x[0] + x[1]*t - y

def jac(x, t, y):
    J = np.empty((t.size, x.size))
    J[:,0] = -np.log(t/x[1] + 1)
    J[:,1] = -x[0]*t/(t*x[1] + x[1]**2)
    J[:,2] = -t
    J[:,3] = -np.exp(x[4]*t)
    J[:,4] = -x[3]*t*np.exp(x[4]*t)
    return J

def main():
    t_min = 0.0
    t_max = 11
    n_points = 8
    t_train = np.array([0.0, 0.1, 0.34, 0.74, 0.96, 4.93, 10.01, 10.92])
    y_train = np.array([260, 100.5, 92.60, 345.9, 95.5, 246.7, 617.7, 262])
   
    x0 = np.array([0.2, 0.5, 0.5, 0.2, 0.2, 0.2])
    x1 = np.array([0.1, 0.0])
    res_lsq = optimize.least_squares(funL, x1, args=(t_train, y_train))

    
    t_test = np.linspace(t_min, t_max, 100)
    y_test = g_data(t_test, *res_lsq.x)
    plt.figure(1)
    plt.plot(t_train, y_train, 'o', label='Experimento')
    plt.plot(t_test, y_test, '--', label='Linear')
    plt.xlabel("i/A")
    plt.ylabel("V")
    plt.legend()
    plt.show()

    tc5 = np.array([1, 4, 7, 10, 11, 13, 14, 17, 20, 24, 31, 45, 52, 56, 59, 61, 63, 67, 70, 73, 77])
    vc5 = np.array([203, 194.6, 164, 95.5, 104, 440.3, 375.7, 103, 105.2, 82, 74, 52.3, 48.8, 28.72, 11.02, 8.78, 8.51, 14.59, 20.54, 25.54, 2.68])
    vc2 = np.array([158, 386, 164, 100.5, 100, 188.3, 55.23, 279.8, 107.8, 66, 61, 46.4, 12.24, 65, -84.1, -134.1, -165.4, -123.2, -50.66])
    vc3 = np.array([178, 150, 248, 345.9, 130, 256.3, 230.8, 186.5, 160, 100.8, 104.2, 46.2, -32.6, 429.7, -53.1, -67.5, -75.8, -122, -82.4])
    vc4 = np.array([275, 92.3, 101.6, 92.6, 211, 264.8, 164.9, 126.4, 264.3, 105.4, 199.7, 229.9, 235, 277.5, 272.2, 235.5, 237.4, 160.7, 102.5, 49.47, 126.9])
    vc6 = np.array([163, 179.1, 347, 617.7, 264, 114.3, 384.2, 579.9, 585, 326.7, 569.6, 236, 105.7, 194.3, 227.3, 195.5, 184, 213, 200.2, 209.9, 184])
    tc8 = np.array([1, 4, 7, 10, 11, 13, 14, 17, 24, 31, 45, 52, 56, 59, 61, 63, 67, 70])
    vc7 = np.array([97, 132, 266, 246.7, 40, 87.5, 75, 79.9, 52.43, 57.8, 74.2, 22, -89, 21.67, 32.98, 65.8, -15.7, -54.61, 2.12])
    vc8 = np.array([321, 225.1, 196.5, 262, 33, 161.2, 165.2, -72.3, 185, 99.2, 0, -9.9, 62.5, 7.2, 4.82, 10.31, 2.13, -77.3])
    
    tdias = np.linspace(tc5[0], tc5[20], 200)
    tc1 = np.array([1, 4, 7, 10, 11, 13, 14, 17, 20, 24, 31, 45, 52, 56, 59, 61, 63, 67, 70])
    vc1 = np.array([112.8, 131, 190, 260, 205, 210.2, 232.2, 249, 23.98, 176, 237.1, 152.7, 115.9, -178.4, 118.1, 130.1, 132.5, 88.5, 101.9])
    
    pcs = interpolate.PchipInterpolator(tc5, vc5)
    print(pcs(1.5))
    plt.figure(2)
    plt.plot(tc5, vc5, 'o', label='data c5')
    plt.plot(tdias, pcs(tdias), label = 'pchi')
    plt.xlabel("dias")
    plt.ylabel("V")
    plt.legend()
    plt.show()

    tdias = np.linspace(1, 70, 200)
    pcs = interpolate.PchipInterpolator(tc1, vc1, axis = 0, extrapolate=None)
    plt.figure(3)
    plt.plot(tc1, vc1, 'o', label='data c1')
    plt.plot(tdias, pcs(tdias), label = 'pchi')
    plt.xlabel("dias")
    plt.ylabel("V")
    plt.legend()
    plt.show()
    
    tdias = np.linspace(1, 70, 200)
    pcs = interpolate.PchipInterpolator(tc1, vc2, axis = 0, extrapolate=None)
    plt.figure(4)
    plt.plot(tc1, vc2, 'o', label='data c2')
    plt.plot(tdias, pcs(tdias), label = 'pchi')
    plt.xlabel("dias")
    plt.ylabel("V")
    plt.legend()
    plt.show()

    tdias = np.linspace(1, 70, 200)
    pcs = interpolate.PchipInterpolator(tc1, vc3, axis = 0, extrapolate=None)
    plt.figure(5)
    plt.plot(tc1, vc3, 'o', label='data c3')
    plt.plot(tdias, pcs(tdias), label = 'pchi')
    plt.xlabel("dias")
    plt.ylabel("V")
    plt.legend()
    plt.show()

    tdias = np.linspace(1, 77, 200)
    pcs = interpolate.PchipInterpolator(tc5, vc4, axis = 0, extrapolate=None)
    plt.figure(6)
    plt.plot(tc5, vc4, 'o', label='data c4')
    plt.plot(tdias, pcs(tdias), label = 'pchi')
    plt.xlabel("dias")
    plt.ylabel("V")
    plt.legend()
    plt.show()

    tdias = np.linspace(1, 77, 100)
    pcs = interpolate.PchipInterpolator(tc5, vc6, axis = 0, extrapolate=None)
    plt.figure(7)
    plt.plot(tc5, vc6, 'o', label='data c6')
    plt.plot(tdias, pcs(tdias), label = 'pchi')
    plt.xlabel("dias")
    plt.ylabel("V")
    plt.legend()
    plt.show()

    tdias = np.linspace(1, 70, 200)
    pcs = interpolate.PchipInterpolator(tc1, vc7, axis = 0, extrapolate=None)
    plt.figure(8)
    plt.plot(tc1, vc7, 'o', label='data c7')
    plt.plot(tdias, pcs(tdias), label = 'pchi')
    plt.xlabel("dias")
    plt.ylabel("V")
    plt.legend()
    plt.show()
    
    tdias = np.linspace(1, 70, 200)
    pcs = interpolate.PchipInterpolator(tc8, vc8, axis = 0, extrapolate=None)
    plt.figure(9)
    plt.plot(tc8, vc8, 'o', label='data c8')
    plt.plot(tdias, pcs(tdias), label = 'pchi')
    plt.xlabel("dias")
    plt.ylabel("V")
    plt.legend()
    plt.show()
   

main()
    
    
