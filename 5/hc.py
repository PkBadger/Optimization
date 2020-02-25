import numpy
from matplotlib import pyplot as plt
from test_gradient import Mult_gradient

numpy.random.seed(100)

seed = [64,32,16,8,4,2,-3.98]

m = 5
b = 2

std = 3

x = numpy.arange(0,11,1)
X = [x, x**2, x**3, x**4, x**5, x**6]

xForPred = numpy.arange(0,10.1,.1)
XForPred = [xForPred, xForPred**2, xForPred**3, xForPred**4, xForPred**5, xForPred**6]


epsilon = numpy.random.normal(0, std, (len(x))) 

y = m * x + b + epsilon

def hc_function(thetas):
    print(thetas)
    return sum((y - numpy.dot(thetas[0:-1],X) - thetas[-1])**2)

#gradient = Mult_gradient(hc_function, .00000000000036455, .0001, .001)
alphas = numpy.arange(.0000000000001 ,.000000000001, .00000000000001)
#alphas = numpy.arange(.000000000000001 ,.000000000001, .000000000000005)
couter = 0
for alpha in alphas:
    gradient = Mult_gradient(hc_function, alpha, alpha, .001)

    thetas = gradient.gradient(seed[0:7])

    #print(thetas)

    ypredHb = numpy.dot(thetas[0:-1],XForPred) + thetas[-1]
    plt.scatter(x,y)
    plt.plot(xForPred, ypredHb)
    plt.savefig('graph_images/' + str(couter) + '.png')
    plt.close()
    couter = couter + 1
