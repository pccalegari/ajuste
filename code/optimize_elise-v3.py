import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy import interpolate

def gen_data(t, a, b, c, d, e, f=0.6795):
    y = f - a*np.log(t/b + 1.0) - c*t - d*np.exp(e*t)
    return y 

def g_data(t, a, b):
    y = a - b*t
    return y 

def fun(x, t, y):
    return 0.6795 - x[0]*np.log(t/x[1] + 1.0) - x[2]*t - x[3]*np.exp(x[4]*t) - y

def funL(x, t, y):
    return x[0] - x[1]*t - y

def main():


    tc1 = np.array([0, 5, 6, 10, 13, 15, 18, 21, 24, 25, 27, 28, 31, 38, 45, 59, 66, 73, 75, 77, 84])
    vc1 = np.array([0, 0.17, 0.1, 0.21, 0.95, 112.8, 131, 190, 260, 205, 210.2, 232.2, 249, 176, 237.1, 152.7, 115.9, 118.1, 130.1, 132.5, 101.9])
    tc2 = np.array([0, 5, 6, 10, 13, 15, 18, 21, 24, 25, 27, 31, 34, 38, 45, 59, 70, 73, 75, 77, 81, 84])
    vc2 = np.array([69, 127.8, 0.106, 95.7, 135, 158, 386, 164, 100.5, 100, 188.3, 279.8, 107.8, 66, 61, 46.4, 65, -84.1, -134.1, -165.4, -123.2, -50.66])
    tc3 = np.array([0, 5, 6, 10, 13, 15, 18, 21, 24, 25, 27, 28, 31, 34, 38, 45, 59, 66, 73, 75, 77, 81, 84])    
    vc3 = np.array([1, 0.9, 0.94, -0.69, 1.01, 178, 150, 248, 345.9, 130, 256.3, 230.8, 186.5, 160, 100.8, 104.2, 46.2, -32.6, -53.1, -67.5, -75.8, -122, -82.4])
    tc4 = np.array([0, 5, 6, 10, 13, 15, 18, 21, 24, 25, 27, 28, 31, 34, 38, 45, 59, 66, 70, 73, 75, 77, 81, 84, 91])
    vc4 = np.array([-2, 0.48, 0.53, 0.77, 1.74, 275, 92.3, 101.6, 92.6, 211, 264.8, 164.9, 126.4, 264.3, 105.4, 199.7, 229.9, 235, 277.5, 272.2, 235.5, 237.4, 160.7, 102.5, 126.9])
    tc5 = np.array([0, 5, 6, 10, 13, 15, 18, 21, 24, 25, 27, 28, 31, 34, 38, 45, 59, 66, 70, 73, 75, 77, 81, 84, 87, 91])
    vc5 = np.array([0.2, 0, 0, 0.11, 0.09, 203, 194.6, 164, 95.5, 104, 440.3, 375.7, 103, 105.2, 82, 74, 52.3, 48.8, 28.72, 11.02, 8.78, 8.51, 14.59, 20.54, 25.54, 2.68])
    tc6 = np.array([0, 5, 6, 10, 13, 15, 18, 21, 24, 25, 28, 31, 34, 45, 59, 70, 73, 75, 77, 81, 84, 87, 91])
    vc6 = np.array([0.32, -0.07, -0.09, 0, 0.36, 163, 179.1, 347, 617.7, 264, 384.2, 579.9, 585, 569.6, 236, 194.3, 227.3, 195.5, 184, 213, 200.2, 209.9, 184])
    tc7 = np.array([0, 5, 6, 10, 13, 15, 18, 21, 24, 27, 28, 31, 34, 38, 45, 59, 70, 73, 75, 77, 81, 84]) 
    vc7 = np.array([0.05, -0.11, -0.12, -0.14, 0.06, 97, 132, 266, 246.7, 87.5, 75, 79.9, 52.43, 57.8, 74.2, 22, 21.67, 32.98, 65.8, -15.7, -54.61, 2.12])
    tc8 = np.array([0, 5, 6, 10, 13, 15, 18, 21, 24, 27, 28, 38, 45, 59, 66, 70, 73,75, 77, 81, 84])
    vc8 = np.array([0.05, 0, 0, 0, 0.1, 321, 225.1, 196.5, 262, 161.2, 165.2, 185, 99.2, 0, -9.9, 62.5, 7.2, 4.82, 10.31, 2.13, -77.3])
    
    tdias = np.linspace(tc1[0], tc1[len(tc1)-1], 200)
    pcs1 = interpolate.PchipInterpolator(tc1, vc1, axis = 0, extrapolate=None)
    plt.figure(1)
    plt.plot(tc1, vc1, 'o', label='Data CCM1')
    plt.plot(tdias, pcs1(tdias), label = 'PCHIP Interpolation')
    plt.xlabel("dias")
    plt.ylabel("V")
    plt.legend()
    plt.show()
    
    tdias = np.linspace(tc2[0], tc2[len(tc2)-1], 200)
    pcs2 = interpolate.PchipInterpolator(tc2, vc2, axis = 0, extrapolate=None)
    plt.figure(2)
    plt.plot(tc2, vc2, 'o', label='Data CCM2')
    plt.plot(tdias, pcs2(tdias), label = 'PCHIP Interpolation')
    plt.xlabel("dias")
    plt.ylabel("V")
    plt.legend()
    plt.show()

    tdias = np.linspace(tc3[0], tc3[len(tc3)-1], 200)
    pcs3 = interpolate.PchipInterpolator(tc3, vc3, axis = 0, extrapolate=None)
    plt.figure(3)
    plt.plot(tc3, vc3, 'o', label='Data CCM3')
    plt.plot(tdias, pcs3(tdias), label = 'PCHIP Interpolation')
    plt.xlabel("dias")
    plt.ylabel("V")
    plt.legend()
    plt.show()

    tdias = np.linspace(tc4[0], tc4[len(tc4)-1], 200)
    pcs4 = interpolate.PchipInterpolator(tc4, vc4, axis = 0, extrapolate=None)
    plt.figure(4)
    plt.plot(tc4, vc4, 'o', label='Data CCM4')
    plt.plot(tdias, pcs4(tdias), label = 'PCHIP Interpolation')
    plt.xlabel("dias")
    plt.ylabel("V")
    plt.legend()
    plt.show()

    tdias = np.linspace(tc5[0], tc5[len(tc5)-1], 200)
    pcs5 = interpolate.PchipInterpolator(tc5, vc5)
    plt.figure(5)
    plt.plot(tc5, vc5, 'o', label='Data CCM5')
    plt.plot(tdias, pcs5(tdias), label = 'PCHIP Interpolation')
    plt.xlabel("dias")
    plt.ylabel("V")
    plt.legend()
    plt.show()

    tdias = np.linspace(tc6[0], tc6[len(tc6)-1], 100)
    pcs6 = interpolate.PchipInterpolator(tc6, vc6, axis = 0, extrapolate=None)
    plt.figure(6)
    plt.plot(tc6, vc6, 'o', label='Data CCM6')
    plt.plot(tdias, pcs6(tdias), label = 'PCHIP Interpolation')
    plt.xlabel("dias")
    plt.ylabel("V")
    plt.legend()
    plt.show()

    tdias = np.linspace(tc7[0], tc7[len(tc7)-1], 200)
    pcs7 = interpolate.PchipInterpolator(tc7, vc7, axis = 0, extrapolate=None)
    plt.figure(7)
    plt.plot(tc7, vc7, 'o', label='Data CCM7')
    plt.plot(tdias, pcs7(tdias), label = 'PCHIP Interpolation')
    plt.xlabel("dias")
    plt.ylabel("V")
    plt.legend()
    plt.show()
    
    tdias = np.linspace(tc8[0], tc8[len(tc8)-1], 200)
    pcs8 = interpolate.PchipInterpolator(tc8, vc8, axis = 0, extrapolate=None)
    plt.figure(8)
    plt.plot(tc8, vc8, 'o', label='Data CCM8')
    plt.plot(tdias, pcs8(tdias), label = 'PCHIP Interpolation')
    plt.xlabel("dias")
    plt.ylabel("V")
    plt.legend()
    plt.show()

    #print(pcs1(22),pcs2(22),pcs3(22),pcs4(22),pcs5(22),pcs6(22),pcs7(22),pcs8(22))
    #print(pcs1(27),pcs2(27),pcs3(27),pcs4(27),pcs5(27),pcs6(27),pcs7(27),pcs8(27))
    #rint(pcs1(33),pcs2(33),pcs3(33),pcs4(33),pcs5(33),pcs6(33),pcs7(33),pcs8(33))
    #rint(pcs1(47),pcs2(47),pcs3(47),pcs4(47),pcs5(47),pcs6(47),pcs7(47),pcs8(47))
    #print(pcs1(75),pcs2(75),pcs3(75),pcs4(75),pcs5(75),pcs6(75),pcs7(75),pcs8(75))

    i = 10**(-3)*np.array([0.0, 0.1883, 0.54532, 0.98074, 1.75, 4.403, 5.36423, 6.71667])
    v = np.array([0.6795, 0.1883, 0.2563, 0.2648, 0.0875, 0.4403, 0.33097, 0.1612])
    i_min = i[0]
    i_max = i[len(i)-1]
    x0 = np.array([0.2, 1.5, 0.5, 0.2, 10.0])
    x1 = np.array([0.1, 1.0])
    res_lsq = optimize.least_squares(funL, x1, args=(i, v))
    res_lsqnl = optimize.least_squares(fun, x0, args=(i, v))

    print("Linear 27")
    print(res_lsq)
    print("Nao Linear 27")
    print(res_lsqnl)
    
    ig = np.linspace(i_min, i_max, 100)
    vgl = g_data(ig, *res_lsq.x)
    vgn = gen_data(ig, *res_lsqnl.x)
    plt.figure(9)
    plt.plot(i, v, 'o', label='Experimento')
    plt.plot(ig, vgl, '--', label='Linear')
    plt.plot(ig, vgn, '--', label='Nao Linear')
    plt.xlabel("i")
    plt.ylabel("V")
    plt.legend()
    plt.show()

    ie = 10**(-3)*np.array([0.0, 0.1005, 0.34296, 0.73596, 0.955, 4.934, 10.0113, 10.9167])
    ve = np.array([0.6795, 0.1005, 0.0926, 0.3459, 0.0955, 0.2467, 0.6177, 0.262])
    ie_min = ie[0]
    ie_max = ie[len(i)-1]
    x0 = np.array([0.2, 1.5, 0.5, 0.2, 100.2])
    x1 = np.array([0.1, 1.0])
    res_lsq = optimize.least_squares(funL, x1, args=(ie, ve))
    res_lsqnl = optimize.least_squares(fun, x0, args=(ie, ve))

    print("Linear 27")
    print(res_lsq)
    print("Nao Linear 27")
    print(res_lsqnl)
    
    ig = np.linspace(ie_min, ie_max, 100)
    vgl = g_data(ig, *res_lsq.x)
    vgn = gen_data(ig, *res_lsqnl.x)
    plt.figure(10)
    plt.plot(ie, ve, 'o', label='Experimento')
    plt.plot(ig, vgl, '.-', label='Linear')
    plt.plot(ig, vgn, '--', label='Nao Linear')
    plt.xlabel("i")
    plt.ylabel("V")
    plt.legend()
    plt.show()
 
main()
    
    
