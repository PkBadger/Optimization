import numpy
from matplotlib import pyplot as plt
from test_gradient import Mult_gradient

numpy.random.seed(100)

seed = [1,1,1,1,1,1,1]

m = 5
b = 2

std = 3

x = numpy.arange(0,11,1)
X = [x, x**2, x**3, x**4, x**5, x**6]

xForPred = numpy.arange(0,10.1,.1)
XForPred = [xForPred, xForPred**2, xForPred**3, xForPred**4, xForPred**5, xForPred**6]


epsilon = numpy.random.normal(0, std, (len(x))) 

y = m * x + b + epsilon

def hb_function(thetas):
    print(thetas)
    return sum((y - numpy.dot(thetas[0:-1],X[0:3]) - thetas[-1])**2)

gradient = Mult_gradient(hb_function, .0000001, .0001,1000)

thetas = gradient.gradient(seed[0:4])

print(thetas)

ypredHb = numpy.dot(thetas[0:-1],XForPred[0:3]) + thetas[-1]
plt.scatter(x,y)
plt.plot(xForPred, ypredHb)
plt.show()
