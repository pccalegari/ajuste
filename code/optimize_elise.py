import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

def gen_data(t, a, b, c, d, e, f=679.5*10**(-3), noise=0, n_outliers=0, random_state=0):
    y = f - a*np.log(t/b + 1.0) - c*t - d*np.exp(e*t)
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(t.size)
    outliers = rnd.randint(0, t.size, n_outliers)
    error[outliers] *= 10

    return y + error

def fun(x, t, y):
    return 679.5*10**(-3) - x[0]*np.log(t/x[1] + 1.0) - x[2]*t - x[3]*np.exp(x[4]*t) - y

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
    t_max = 4.0*10**(-3)
    n_points = 8
    t_train = np.array([0.0, 4.75*10**(-4), 8.27*10**(-4), 1.69*10**(-3), 2.5*10**(-3), 3.01*10**(-3), 3.19*10**(-3), 3.86*10**(-3)])
    y_train = np.array([0.6795, 0.4862, 0.423, 0.33975, 0.25075, 0.2062, 0.1629, 0.07715])
    #a = 0.5
    #b = 2.0
    #t_min = 0
    #t_max = 10
    #n_points = 15

    #t_train = np.linspace(t_min, t_max, n_points)
    #y_train = gen_data(t_train, a, b, noise=0.1, n_outliers=3)

    x0 = np.array([0.0, 0.5, 0.0, 0.0, 900.0])

    res_lsq = optimize.least_squares(fun, x0, jac=jac, args=(t_train, y_train))
    res_lsq1 = optimize.least_squares(fun, x0, args=(t_train, y_train))
    res_soft = optimize.least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(t_train, y_train))
    res_log = optimize.least_squares(fun, x0, loss='cauchy', f_scale=0.1, args=(t_train, y_train))
    print("Linear")
    print(res_lsq)
    print("Soft")
    print(res_soft)
    print("Log")
    print(res_log)
    
    t_test = np.linspace(t_min, t_max, 100)
    #y_true = gen_data(t_test, a, b, c)
    y_lsq = gen_data(t_test, *res_lsq.x)
    y_lsq1 = gen_data(t_test, *res_lsq1.x)
    y_soft = gen_data(t_test, *res_soft.x)
    y_log = gen_data(t_test, *res_log.x)

    plt.plot(t_train, y_train, 'o', label='Experimento')
    #plt.plot(t_test, y_true, 'k', linewidth=2, label='true')
    #plt.plot(t_test, y_lsq, label='linear loss')
    plt.plot(t_test, y_lsq1, label='Ajuste')
    #plt.plot(t_test, y_soft, label='soft loss')
    #plt.plot(t_test, y_log, label='cauchy loss')
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.show()

main()
    
    
