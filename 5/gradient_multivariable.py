from sympy import *
import numpy

x,y = symbols('x y')

class Mult_gradient:
    def __init__(self):
        pass

def function(x1,y1):
    x0 = numpy.matrix([1,3])

    values = numpy.matrix([[2,1],[1,20]])

    print(x0.T.shape)
    print(values.shape)

    print(x0.dot(values).dot(x0.T))

if __name__ == "__main__":
    initial_values = numpy.matrix([1,3])
    print(initial_values.shape)
    function(1,3)