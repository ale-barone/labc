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

def dipole(param, q2):
    f0, M = param
    return f0 / (1 + q2/M**2)**2

def monopole(param, q2):
    f0, M = param
    return f0 / (1 + q2/M**2)


################################################################################
# Classes for class Fitter
################################################################################


class Const:
    STRING = 'f(x) = C'
    PARAM = {0: 'C'}
    ARGS = {}

    def __new__(cls):
        return const
    
class Lin:
    STRING = 'f(x) = mx + c'
    PARAM = {0: 'm', 1: 'c'}
    ARGS ={}

    def __new__(cls):
        return lin

class Exp:
    STRING = 'f(t) = \sum_i^N Ai*exp(-Ei*t)'
    PARAM = {0: 'A', 1: 'E'}
    ARGS = {'N'}


    def __new__(cls):
        def _exp(param, t, *, N):
            A = param[:N]
            E = param[N:2*N]
            out = np.sum([Ai*np.exp(-Ei*t) for Ai, Ei in zip(A, E)], axis=0)
            return out
        return _exp
    
    # #@staticmethod
    # @classmethod
    # def get_param(cls):
    #     return cls.ARGS

class ExpTmp:
    STRING = 'f(t) = \sum_i^N Ai*exp(-Ei*t)'
    PARAM = {0: 'A', 1: 'E'}

    def __new__(cls):
        def multiexp(param, t):
            num_state = int(len(param)/2)
            A = param[:num_state]
            E = param[num_state:]

            out = np.sum([Ai*np.exp(-Ei*t) for Ai, Ei in zip(A, E)], axis=0)
            return out
        return multiexp


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
    
class Dipole:
    STRING = 'f(q2) = f0 / (1 +q2/M**2)**2'
    PARAM = {0: 'f0', 1: 'M'}
    ARGS = {}

    def __new__(cls):
        return dipole

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
    

def dipoleZfit(param, q2, *, tcut, nmax=None):
    param_zfit = param[:-1]
    M = param[-1]
    dip = 1 / (1+q2/M**2)**2
    zf = zfit(param_zfit, q2, tcut=tcut, nmax=nmax)

    return dip*zf

class DipoleZfit:
    STRING = ''
    PARAM = {0: 'a', 1: 'b0', 2: 'M'}
    ARGS = {}

    def __new__(cls):
        def _dipoleZfit(param, var, *, tcut, nmax):
          #tcut = (3*0.1348)**2
          q2 = var.T[0]
          ts = var.T[1]
          z = (np.sqrt(tcut+q2)-np.sqrt(tcut)) / (np.sqrt(tcut+q2)+np.sqrt(tcut))

          q2_unique = np.unique(q2)
          num_q2 = len(q2_unique)
          ts_unique = np.unique(ts)
          num_ts = len(ts_unique)

          M = param[-1]
          dip = 1 / (1 + q2/M**2)**2

          a = np.array([param[i] for i in range(nmax+1)])
          b0 = np.array([param[i] for i in range(nmax+1, (nmax+1)+num_q2)])
          b0 = np.array([b0 for i in range(num_ts)]).T.flatten()
          
          zfit = np.sum(a[i]*z**i for i in range(nmax+1))

          out = b0 + ts*dip*zfit
          return out
        return _dipoleZfit
    


def monopoleZfit(param, q2, *, tcut, nmax=None):
    param_zfit = param[:-1]
    M = param[-1]
    mono = 1 / (1+q2/M**2)
    zf = zfit(param_zfit, q2, tcut=tcut, nmax=nmax)

    return mono*zf

class MonopoleZfit:
    STRING = ''
    PARAM = {0: 'a', 1: 'b0', 2: 'M'}
    ARGS = {}

    def __new__(cls):
        def _monopoleZfit(param, var, *, tcut, nmax):
          #tcut = (3*0.1348)**2
          q2 = var.T[0]
          ts = var.T[1]
          z = (np.sqrt(tcut+q2)-np.sqrt(tcut)) / (np.sqrt(tcut+q2)+np.sqrt(tcut))

          q2_unique = np.unique(q2)
          num_q2 = len(q2_unique)
          ts_unique = np.unique(ts)
          num_ts = len(ts_unique)

          M = param[-1]
          monop = 1 / (1 + q2/M**2)

          a = np.array([param[i] for i in range(nmax+1)])
          b0 = np.array([param[i] for i in range(nmax+1, (nmax+1)+num_q2)])
          b0 = np.array([b0 for i in range(num_ts)]).T.flatten()
          
          a = np.array([param[i] for i in range(nmax+1)])
          b0 = np.array([param[i] for i in range(nmax+1, (nmax+1)+num_q2)])
          b0 = np.array([b0 for i in range(num_ts)]).T.flatten()
          
          zfit = np.sum(a[i]*z**i for i in range(nmax+1))

          out = b0 + ts*monop*zfit
          return out
        return _monopoleZfit
    



def ansatz_2(coeff, Mpi, *, Mn=0.93892, Fpi=0.09242, L=None):
            ga_0 = coeff[0]
            d16 = coeff[1]
            deltac2c3 = coeff[2]
            
            ga_1 = 4*d16 - (ga_0)**3 / (16*np.pi**2*Fpi**2)
            ga_2 = ga_0*(1+2*ga_0**2) / (8 * np.pi**2 * Fpi**2) 
            ga_3 = ga_0*(1+ga_0**2) / (8*np.pi**2*Fpi**2*Mn) - ga_0*deltac2c3/(6*np.pi*Fpi**2)
            
            a0 = ga_0 + ga_1*Mpi**2 - ga_2*Mpi**2*np.log(Mpi/Mn) + ga_3*Mpi**3
            if L is not None:
                a0 += Mpi**2*np.exp(-Mpi*L)/np.sqrt(Mpi*L)
            return a0

class Zfita0:
    STRING = ''
    PARAM = {0: 'ga_0', 1: 'd16', 2: 'deltac2c3'}
    
    def __new__(cls):
        return ansatz_2
    

def ansatzcont_2(coeff, Mpia, *, Mn=0.93892, Fpi=0.09242, L=None):
    Mpi = Mpia.T[0]
    a = Mpia.T[1]

    ga_0 = coeff[0]
    d16 = coeff[1]
    deltac2c3 = coeff[2]
    a0_a2_0 = coeff[3]
    
    ga_1 = 4*d16 - (ga_0)**3 / (16*np.pi**2*Fpi**2)
    ga_2 = ga_0*(1+2*ga_0**2) / (8 * np.pi**2 * Fpi**2) 
    ga_3 = ga_0*(1+ga_0**2) / (8*np.pi**2*Fpi**2*Mn) - ga_0*deltac2c3/(6*np.pi*Fpi**2)
    
    a0 = ga_0 + ga_1*Mpi**2 - ga_2*Mpi**2*np.log(Mpi/Mn) + ga_3*Mpi**3 + a0_a2_0*a**2
    if L is not None:
        a0 += Mpi**2*np.exp(-Mpi*L)/np.sqrt(Mpi*L)
    return a0

class Zfita0cont:
    STRING = ''
    PARAM = {0: 'ga_0', 1: 'd16', 2: 'deltac2c3', 3: 'a0_a2_0'}
    
    def __new__(cls):
        return ansatzcont_2


def ansatz_linear(coeff, Mpia, *, Mn=0.93892, Fpi=0.09242, L=None):
    Mpi = Mpia.T[0]
    a = Mpia.T[1]

    c0 = coeff[0]
    c1 = coeff[1]
    c2 = coeff[2]

    return c0 + c1*Mpi**2 + c2*a**2

class ContExtr:
    STRING = ''
    PARAM = {0: 'c0', 1: 'c1', 2: 'c2'}
    
    def __new__(cls):
        return ansatz_linear



class TmpFit:
    STRING = ''
    PARAM = {0: 'c0', 1: 'c1', 2: 'c2'}

    def __new__(cls):
        def tmpfit(coeff, Mpia, *, Mn=0.93892, Fpi=0.09242, L=None):
            Mpi = Mpia.T[0][::3]
            a = Mpia.T[1][::3]
            
            coeffa0 = coeff[:4]
            coeffa1 = coeff[4:7]
            coeffa2 = coeff[7:10]

            ga_0 = coeffa0[0]
            d16 = coeffa0[1]
            deltac2c3 = coeffa0[2]
            a0_a2_0 = coeffa0[3]
            
            ga_1 = 4*d16 - (ga_0)**3 / (16*np.pi**2*Fpi**2)
            ga_2 = ga_0*(1+2*ga_0**2) / (8 * np.pi**2 * Fpi**2) 
            ga_3 = ga_0*(1+ga_0**2) / (8*np.pi**2*Fpi**2*Mn) - ga_0*deltac2c3/(6*np.pi*Fpi**2)
            
            a0 = ga_0 + ga_1*Mpi**2 - ga_2*Mpi**2*np.log(Mpi/Mn) + ga_3*Mpi**3 + a0_a2_0*a**2

            a1 = coeffa1[0] + coeffa1[1]*Mpi**2 + coeffa1[2]*a**2
            a2 = coeffa2[0] + coeffa2[1]*Mpi**2 + coeffa2[2]*a**2

            return np.array([a0, a1, a2]).T
        return tmpfit
    
class ContExtrLin:
    STRING = ''
    PARAM = {0: 'c0', 1: 'c1', 2: 'c2'}

    def __new__(cls):
        def tmpfit(coeff, Mpia, *, Mn=0.93892, Fpi=0.09242, L=None):
            Mpi = Mpia.T[0][::3]
            a = Mpia.T[1][::3]
            
            coeffa0 = coeff[:3]
            coeffa1 = coeff[3:6]
            coeffa2 = coeff[6:9]
            
            a0 = coeffa0[0] + coeffa0[1]*Mpi**2 + coeffa0[2]*a**2
            a1 = coeffa1[0] + coeffa1[1]*Mpi**2 + coeffa1[2]*a**2
            a2 = coeffa2[0] + coeffa2[1]*Mpi**2 + coeffa2[2]*a**2

            return np.array([a0, a1, a2]).T
        return tmpfit
