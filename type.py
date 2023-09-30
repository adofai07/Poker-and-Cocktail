import decimal
import functools
import numpy as np

decimal.getcontext().prec = 100
D = decimal.Decimal

class DecimalComplex:
    def __init__(self, real: D, imag: D):
        self.real = real
        self.imag = imag

    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag

    def __add__(self, other):
        return DecimalComplex(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        return DecimalComplex(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        return DecimalComplex(self.real * other.real - self.imag * other.imag, self.real * other.imag + self.imag * other.real)
    
    def __pow__(self, other):
        ret = DecimalComplex(D(1), D(0))
        base = self
        exp = other

        while True:
            if exp & 1:
                ret *= base
            
            exp >>= 1

            if exp == 0:
                return ret
            
            base *= base

    def reciprocal(self):
        return DecimalComplex(self.real / self.norm() ** 2, -self.imag / self.norm() ** 2)
    
    def norm(self):
        return (self.real ** 2 + self.imag ** 2).sqrt()

    def __str__(self):
        return F"{self.real} + {self.imag}i"

@functools.cache
def fact(x: int) -> int:
    if x == 0: return 1
    if x == 1: return 1

    return x * fact(x - 1)

@functools.cache
def comb(n: int, k: int) -> int:
    if n - k < k: return comb(n, n - k)

    if n == 1: return 1
    if k == 0: return 1

    return comb(n - 1, k - 1) + comb(n - 1, k)

def p(K: int, m: int, l: int) -> D:
    assert K % 2 == 1

    if l == m:
        return 1 - (
            D(((K + 1) // 2) ** m - (K // 2) ** m - 1) /
            D(K ** (m - 1))
        )

    else:
        return (
            D(comb(m, l) * (K // 2) ** (m - l)) / 
            D(K ** (m - 1))
        )

@functools.cache
def ExpectedStoppingTimeWithXPlayers(X: int, K: int) -> D:
    if X == 1: return D(0)

    return (D(1) + sum(ExpectedStoppingTimeWithXPlayers(i, K) * p(K, X, i) for i in range(1, X))) / (D(1) - p(K, X, X))

def DecimalPrecisionExponentialGeneratingFunctionOfStoppingTime(x: DecimalComplex, K: int, iters=300) -> DecimalComplex:
    ret = DecimalComplex(D(0), D(0))

    for i in range(2, iters + 2):
        ret += (
            DecimalComplex(ExpectedStoppingTimeWithXPlayers(i, K), D(0)) * 
            x ** i * 
            DecimalComplex(D(fact(i)), D(0)).reciprocal()
        )
        
    return ret

def DecimalPrecisionOrdinaryPowerSeriesGeneratingFunctionOfStoppingTime(x: DecimalComplex, K: int, iters=300) -> DecimalComplex:
    threshold = D(1.79e308)
    ret = DecimalComplex(D(0), D(0))

    for i in range(2, iters + 2):
        ret += (
            DecimalComplex(ExpectedStoppingTimeWithXPlayers(i, K), D(0)) * 
            x ** i
        )

        if ret.norm() > threshold:
            break

    if ret.norm() > threshold:
        return np.nan

    return ret

def DoublePrecisionExponentialGeneratingFunctionOfStoppingTime(x: complex, K: int, iters=300) -> complex:
    res = DecimalPrecisionExponentialGeneratingFunctionOfStoppingTime(DecimalComplex(D(x.real), D(x.imag)), K, iters)

    return float(res.real) + float(res.imag) * 1j

def DoublePrecisionOrdinaryPowerSeriesGeneratingFunctionOfStoppingTime(x: complex, K: int, iters=300) -> complex:
    res = DecimalPrecisionOrdinaryPowerSeriesGeneratingFunctionOfStoppingTime(DecimalComplex(D(x.real), D(x.imag)), K, iters)

    if res == np.nan:
        return np.nan

    return float(res.real) + float(res.imag) * 1j

if __name__ == "__main__":
    a = D(10 ** 1000)

    print(float(a))