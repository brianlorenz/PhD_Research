"""
This module solves the Lane-Emden equations for polytropes of different n <= 4.5.
The solution is then plotted, and stored as ascii in the local '/output/' directory.

Author: Ehsan Moravveji
Date: 8 Oct 2014

https://bitbucket.org/ehsan_moravveji/ivs_sse/src/master/exercises/exercise07/polytrope.py

"""

import sys
import os
import glob
import numpy as np
import pylab as plt


def write_as_ascii(list_dic):
    """
    Write all results as text file
    """
    n_dic = len(list_dic)
    if not os.path.exists('output'):
        os.mkdir('output')

    for i_dic, dic in enumerate(list_dic):
        filename = 'output/Poly-n{0}.txt'.format(dic['n'])
        list_col = []
        xi = dic['xi']
        theta = dic['theta']
        d_theta = dic['d_theta']
        m = len(xi)
        for i in range(m):
            txt = '{0:016.12f}  {1:016.12f}  {2:016.12f} \n'.format(
                xi[i], theta[i], d_theta[i])
            list_col.append(txt)
        with open(filename, 'w') as w:
            w.writelines(list_col)

    return None


def plot_theta(list_dic):
    """
    Plot all polytropic solutions
    """
    import itertools

    n_dic = len(list_dic)
    fig, (top, bot) = plt.subplots(2, 1, figsize=(6, 8), dpi=200)
    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.09,
                        top=0.98, wspace=0.04, hspace=0.04)
    cyc_col = itertools.cycle(
        ['orange', 'blue', 'red', 'green', 'cyan', 'grey', 'purple', 'pink'])
    cyc_ls = itertools.cycle(['solid'])
    lis_col = list(itertools.islice(cyc_col, len(list_dic)))
    lis_ls = list(itertools.islice(cyc_ls, len(list_dic)))

    for i_dic, dic in enumerate(list_dic):
        xi = dic['xi']
        theta = dic['theta']
        d_theta = dic['d_theta']

        ls = lis_ls[i_dic]
        cl = lis_col[i_dic]
        lbl = r'n=' + str(dic['n'])

        top.plot(xi, theta, linestyle=ls, color=cl, lw=2, label=lbl)
        bot.plot(xi, d_theta, linestyle=ls, color=cl, lw=2)

    top.set_xlim(-0.5, 15)
    top.set_xticklabels(())
    top.set_ylim(-0.1, 1.1)
    top.set_ylabel(r'$\theta(\xi)$')
    bot.set_xlim(-0.5, 15)
    bot.set_xlabel(r'Dimensionless Radius $\xi$')
    bot.set_ylim(-0.6, 0.05)
    bot.set_ylabel(r'$d\theta(\xi)\,/\,d\xi$')

    leg1 = top.legend(loc=1)

    plt.savefig('Lane-Emden.png')
    print(' New plot: "Lane-Emden.png" created')
    plt.close()

    return None


def get_next_func(step, func, deriv):
    """
    (theta)i+1 = (theta)i + d_xi *[ (d_theta/d_xi)i+1].
    """
    return func + step * deriv


def get_next_deriv(xi, step, n, deriv, func):
    """
    (d_theta/d_xi)i+1 = (d_theta/d_xi)i - ([ (2 /xi)i . (d_theta/d_xi)i ] + theta^n ) * d_xi
    """
    return deriv - step * ((2./xi)*deriv + func**n)


def integrate(n=1.0):
    """
    integrate the Lane Emden equations, and return the solution as a dictionary
    """

    if n > 4.99:
        raise SystemExit('Error: integrate: choose n < 4.99.')

    list_xi = []
    list_theta = []
    list_d_theta = []

    # Initial Conditions
    xi = 1e-6
    theta = 1.0
    d_theta = 0.0

    # Stepsize
    xi_step = 1e-3

    i = 0
    list_xi.append(xi)
    list_theta.append(theta)
    list_d_theta.append(d_theta)

    next_theta = theta

    while next_theta >= 0.0:
        next_d_theta = get_next_deriv(xi, xi_step, n, d_theta, theta)
        next_theta = get_next_func(xi_step, theta, next_d_theta)

        if (next_theta > theta):
            print('theta increasing: ', i, xi, theta, next_theta, next_d_theta)

        i += 1
        xi += xi_step
        theta = next_theta
        d_theta = next_d_theta
        list_xi.append(xi)
        list_theta.append(next_theta)
        list_d_theta.append(next_d_theta)

    print(' Solution for n={0} found after {1} steps'.format(n, i))
    list_xi = np.array(list_xi)
    list_theta = np.array(list_theta)
    list_d_theta = np.array(list_d_theta)
    dic = {'n': n, 'xi': list_xi, 'theta': list_theta, 'd_theta': list_d_theta}

    return dic


def main():
    """
    The main caller
    """
    n_arr = [1.5, 3]
    list_dic = []

    for n in n_arr:
        list_dic.append(integrate(n))

    plot_theta(list_dic)

    write_as_ascii(list_dic)

    return None


if __name__ == '__main__':
    status = main()
    sys.exit(status)
