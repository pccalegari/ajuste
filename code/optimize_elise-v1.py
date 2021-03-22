import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

def gen_data(t, a, b, c, d, e, f=190.0, noise=0, n_outliers=0, random_state=0):
    y = f - a*np.log(t/b + 1.0) - c*t - d*np.exp(e*t)
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(t.size)
    outliers = rnd.randint(0, t.size, n_outliers)
    error[outliers] *= 10

    return y + error

def fun(x, t, y):
    return 190.0 - x[0]*np.log(t/x[1] + 1.0) - x[2]*t - x[3]*np.exp(x[4]*t) - y

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
    t_max = 10
    n_points = 8
    i4 = np.array([0.0, 0.164, 0.3762963, 0.52765957, 1.64, 5.32, 5.62398703, 8.1875])
    v4 = np.array([190, 164, 101.6, 248, 164, 266, 347, 196.5])

    i7 = np.array([0.0, 0.1005, 0.34296, 0.73596, 0.955, 4.934, 10.0113, 10.9167])
    v7 = np.array([260, 100.5, 92.6, 345.9, 95.5, 246.7, 617.7, 262])

    i10 = np.array([0.0, 0.1883, 0.54532, 0.98074, 1.75, 1.85251, 4.403, 6.71667])
    v10 = np.array([210.2, 188.3, 256.3, 264.8, 87.5, 114.3, 440.3, 161.2])

    i11 = np.array([0, 0.05523, 0.49106, 0.61074, 1.5, 3.757, 6.2269, 6.88333])
    v11 = np.array([232.2, 55.23, 230.8, 164.9, 75, 375.7, 384.2, 165.2])
   
    x0 = np.array([0.5, 0.5, 0.5, 0.5, 190.0])

    #res_lsq = optimize.least_squares(fun, x0, jac=jac, args=(t_train, y_train))
    #res_lsq1 = optimize.least_squares(fun, x0, args=(i4, v4))
    #res_soft = optimize.least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(t_train, y_train))
    #res_log = optimize.least_squares(fun, x0, loss='cauchy', f_scale=0.1, args=(t_train, y_train))
    #print("Linear")
    #print(res_lsq)
    #print("Soft")
    #print(res_soft)
    #print("Log")
    #print(res_log)
    
    t_test = np.linspace(t_min, t_max, 100)
    #y_true = gen_data(t_test, a, b, c)
    #y_lsq = gen_data(t_test, *res_lsq.x)
   # y_lsq1 = gen_data(t_test, *res_lsq1.x)
   # y_soft = gen_data(t_test, *res_soft.x)
   # y_log = gen_data(t_test, *res_log.x)

    plt.plot(i4, v4, 'o-', label='4 dez')
    plt.plot(i7, v7, '*-', label='7 dez')
    plt.plot(i10, v10, 's-', label='10 dez')
    plt.plot(i11, v11, '.-', label='11 dez')
    #plt.plot(t_test, y_soft, label='soft loss')
    #plt.plot(t_test, y_log, label='cauchy loss')
    plt.xlabel("i")
    plt.ylabel("v")
    plt.legend()
    plt.show()

main()
    
    
