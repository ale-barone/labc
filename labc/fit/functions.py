import numpy as np


def const(param, t):
    return param[0]

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
