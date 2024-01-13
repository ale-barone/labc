import numpy as np


def const(param, t):
    return param[0]

def lin(param, t):
    return param[0]*t + param[1]

def exp(param, t):
    return param[0] * np.exp(-param[1]*t)

def exp2(param, t):
    A0, A1 = param[0]
    E0, E1 = param[1]
    return A0 * np.exp(-E0*t) + A1 * np.exp(-E1*t)

def exp2(param, t):
    A0 = param[0]
    A1 = param[1]
    E0 = param[2]
    E1 = param[3]
    return A0 * np.exp(-E0*t) + A1 * np.exp(-E1*t)

def cosh(p: np.ndarray, t: np.ndarray, T: int) -> np.ndarray:
    Thalf = T/2
    ampl = 2 * p[0]
    exp = np.exp(-p[1]*Thalf)
    cosh = np.cosh(p[1]*(Thalf - t)) 
    return ampl * exp * cosh

def pole(param, t, M):
    return param[0] / (M-t)



################################################################################
# Classes for class Fitter
################################################################################


class Const:
    STRING = 'f(t) = C'
    PARAM = {0: 'C'}
    ARGS = {}

    def __new__(cls):
        return const
    
class Lin:
    STRING = 'f(t) = Ax + B'
    PARAM = {0: 'A', 1: 'B'}
    ARGS ={}

    def __new__(cls):
        return lin

class Exp:
    STRING = 'f(t) = A*exp(-E*t)'
    PARAM = {0: 'A', 1: 'E'}
    ARGS = {}

    def __new__(cls):
        return exp
    
class Exp2:
    STRING = 'f(t) = \sum_i Ai*exp(-Ei*t)'
    PARAM = {0: 'A0', 1: 'A1', 2: 'E0', 3: 'E1'}
    ARGS = {}

    def __new__(cls):
        return exp2
    
class Cosh:
    STRING = 'f(t) = 2A*exp(-E*T/2) * cosh[E*(T/2-t)]'
    PARAM = {0: 'A', 1: 'E'}
    ARGS = {'T'}

    def __new__(cls):
        return cosh

class Pole:
    STRING = 'f(t) = A / (M-t)'
    PARAM = {0: 'A'}
    ARGS = {'M'}

    def __new__(cls):
        return pole


def zfit2(param, q2, *, tcut):
    z = (np.sqrt(tcut+q2)-np.sqrt(tcut)) / (np.sqrt(tcut+q2)+np.sqrt(tcut))
    return param[0] + param[1]*z + param[2]*z**2

class Zfit2:
    STRING = 'f(q2) = a0 + a1*z(q2) + a2*z^2(q2), with '
    PARAM = {0: 'a0', 1: 'a1', 2: 'a2'}
    ARGS = {'tcut'}

    def __new__(cls):
        return zfit2


# def linzfit2(param, var, tcut=None):
#     q2 = var.T[0]
#     ts = var.T[1]
#     z = (np.sqrt(tcut+q2)-np.sqrt(tcut)) / (np.sqrt(tcut+q2)+np.sqrt(tcut))
#     # def b0(q2):
#     #     ll = np.arange(len(q2))

#     #     return param[0]()
#     # ll = np.arange(len(q2))
#     b0 = np.array([param[i] for i in range(20)])
#     b0 = np.array([b0 for i in range(6)]).T.flatten()
#     out = b0 + ts*(param[20] + param[21]*z + param[22]*z**2)
#     return out

# class LinZfit2:
#     STRING = ''
#     PARAM_b0 = {i: f'b0{i}' for i in range(20)}
#     PARAM = {**PARAM_b0, 20: 'a0', 21: 'a1', 22: 'a2'}
#     ARGS = {}

#     def __new__(cls):
#         return linzfit2


def zfit(param, q2, *, tcut, nmax=None):
    if nmax is None:
        nmax = len(param)-1
    z = (np.sqrt(tcut+q2)-np.sqrt(tcut)) / (np.sqrt(tcut+q2)+np.sqrt(tcut))
    return np.sum(param[i]*z**i for i in range(nmax+1))

class Zfit:
    STRING = 'f(q2) = a0 + a1*z(q2) + a2*z^2(q2), with '
    PARAM = {0: 'a'}
    ARGS = {'tcut'}

    def __new__(cls):
        return zfit

def directzfit(param, var, *, tcut, nmax):
    #tcut = (3*0.1348)**2
    q2 = var.T[0]
    ts = var.T[1]
    z = (np.sqrt(tcut+q2)-np.sqrt(tcut)) / (np.sqrt(tcut+q2)+np.sqrt(tcut))

    q2_unique = np.unique(q2)
    num_q2 = len(q2_unique)
    ts_unique = np.unique(ts)
    num_ts = len(ts_unique)

    a = np.array([param[i] for i in range(nmax+1)])
    b0 = np.array([param[i] for i in range(nmax+1, (nmax+1)+num_q2)])
    b0 = np.array([b0 for i in range(num_ts)]).T.flatten()

    out = b0 + ts*np.sum(a[i]*z**i for i in range(nmax+1))
    return out

class FFdirectZfit:
    STRING = ''
    PARAM = {0: 'a', 1: 'b0'}
    ARGS = {}

    def __new__(cls):
        return directzfit


def directzfitb0(param, var, *, tcut, nmax_a, nmax_b):
    #tcut = (3*0.1348)**2
    q2 = var.T[0]
    ts = var.T[1]
    
    za = (np.sqrt(tcut+q2)-np.sqrt(tcut)) / (np.sqrt(tcut+q2)+np.sqrt(tcut))
    tcutb = tcut
    zb = (np.sqrt(tcutb+q2)-np.sqrt(tcutb)) / (np.sqrt(tcutb+q2)+np.sqrt(tcutb))

    a = np.array([param[i] for i in range(nmax_a+1)])
    b0 = np.array([param[i] for i in range(nmax_a+1, (nmax_a+1)+(nmax_b+1))])

    out_a = ts*np.sum(a[i]*za**i for i in range(nmax_a+1))
    out_b = np.sum(b0[i]*zb**i for i in range(nmax_b+1)) 
    return out_a + out_b

class FFdirectZfitb0:
    STRING = ''
    PARAM = {0: 'a', 1: 'b0'}
    ARGS = {}

    def __new__(cls):
        return directzfitb0
