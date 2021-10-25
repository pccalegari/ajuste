import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

def gen_data(t, a, b, c, d, e, f=679.5*10**(-3), noise=0, n_outliers=0, random_state=0):
    y = f - a*np.log(t/b + 1.0) - c*t - d*np.exp(e*t)
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(t.size)
    outliers = rnd.randint(0, t.size, n_outliers)
    error[outliers] *= 10

    return y

def gen_dataL(t, a, b):
    y = a - b*t 
    return y

def fun(x, t, y):
    return 679.5*10**(-3) - x[0]*np.log(t/x[1] + 1.0) - x[2]*t - x[3]*np.exp(x[4]*t) - y

def funL(x, t, y):
    return x[0] - x[1]*t - y

def jac(x, t, y):
    J = np.empty((t.size, x.size))
    J[:,0] = -np.log(t/x[1] + 1)
    J[:,1] = -x[0]*t/(t*x[1] + x[1]**2)
    J[:,2] = -t
    J[:,3] = -np.exp(x[4]*t)
    J[:,4] = -x[3]*t*np.exp(x[4]*t)
    return J

def main():
    i_min = 0.0
    i_max = 4.0*10**(-3)
    n_points = 8
    
    i_train = np.array([0.0, 4.75*10**(-4), 8.27*10**(-4), 1.69*10**(-3), 2.5*10**(-3), 3.01*10**(-3), 3.19*10**(-3), 3.86*10**(-3)])
    i_a = np.array([0, 0.069307, 0.115983, 0.240453, 0.353607, 0.427157, 0.452617, 0.544554])
    v_train = np.array([0.6795, 0.4862, 0.423, 0.33975, 0.25075, 0.2062, 0.1629, 0.07715])
    

    x0 = np.array([0.0, 0.1, 100, 0.0, 500.0])
    xi0 = np.array([0.1, 0.1])

    res_lsq = optimize.least_squares(fun, x0, jac=jac, args=(i_train, v_train))
    res_lsq1 = optimize.least_squares(fun, x0, args=(i_train, v_train))
    res_lin = optimize.least_squares(funL, xi0, args=(i_train, v_train))
    res_log = optimize.least_squares(fun, x0, loss='cauchy', f_scale=0.1, args=(i_train, v_train))
    print("Nao linear")
    print(res_lsq1)
    print("Linear")
    print(res_lin)
    #print("Log")
    #print(res_log)
    
    i_test = np.linspace(i_min, i_max, 100)
    #y_true = gen_data(t_test, a, b, c)
    y_lsq = gen_data(i_test, *res_lsq.x)
    y_lsq1 = gen_data(i_test, *res_lsq1.x)
    y_lin = gen_dataL(i_test, *res_lin.x)
    y_log = gen_data(i_test, *res_log.x)

    plt.figure(1)
    plt.plot(i_train, v_train, 'o', label='Experimento')
    #plt.plot(t_test, y_true, 'k', linewidth=2, label='true')
    #plt.plot(i_test, y_lsq, label='linear loss')
    plt.plot(i_test, y_lsq1, label='Ajuste nao linear')
    #plt.plot(t_test, y_soft, label='soft loss')
    #plt.plot(i_test, y_lin, label='Ajuste linear')
    plt.xlabel("Corrente")
    plt.ylabel("Voltagem")
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(i_train, v_train, 'o', label='Experimento')
    #plt.plot(t_test, y_true, 'k', linewidth=2, label='true')
    #plt.plot(i_test, y_lsq, label='linear loss')
    #plt.plot(i_test, y_lsq1, label='Ajuste nao linear')
    #plt.plot(t_test, y_soft, label='soft loss')
    plt.plot(i_test, y_lin, label='Ajuste linear')
    plt.xlabel("Corrente")
    plt.ylabel("Voltagem")
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.plot(i_a, v_train, '-.')
    plt.xlabel("Densidade de corrente")
    plt.ylabel("Voltagem")
    plt.legend()
    plt.show()

main()
    
    
