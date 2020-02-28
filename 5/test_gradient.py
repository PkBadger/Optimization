import numpy
from matplotlib import pyplot as plt


class Mult_gradient:
    def __init__(self, fx, alpha, h, treshold):
        self.alpha = alpha
        self.h = h
        self.treshold = treshold
        self.function = fx

    def gradient(self, initial_values):
        initial_values = numpy.matrix(initial_values).T
        counter = 0
        while True:
            derivative = self.calculate_derivative(initial_values)
            initial_values = initial_values - self.alpha * derivative
            if(abs(numpy.max(derivative)) < self.treshold): break
            if counter > 70000: 
                break
            counter = counter + 1
        return numpy.asarray(initial_values.T)[0]
    
    def calculate_derivative(self, initial_values):
        fx = self._function(initial_values)
        diagonal = numpy.diag(numpy.full(initial_values.shape[0], self.h))

        inputs = initial_values.T + diagonal

        return numpy.matrix([[self.derivative(fx, inp.T)] for inp in inputs])

    def derivative(self, fx, fxh):
        return (self._function(fxh) - fx)/self.h

    def _function(self, values):
        values = numpy.asarray(values.T)[0]
        return self.function(values)


if __name__ == "__main__":
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

    #plt.scatter(x,y)

    #plt.show()

    ##### EJERCICIO 1 ####
    def l_function(thetas):
        print(sum((y - numpy.dot(thetas[0:-1],X[0:1]) - thetas[-1])**2))
        return sum((y - numpy.dot(thetas[0:-1],X[0:1]) - thetas[-1])**2)

    #res = minimize(l_function, seed[0:2], method='BFGS')

    gradient = Mult_gradient(l_function, .001, .00001, .00000001)

    thetas = gradient.gradient(seed[0:2])

    print(thetas)

    ypred = thetas[0] * x + thetas[1] 
    plt.scatter(x,y)
    plt.plot(ypred)
    plt.show()


    def ha_function(thetas):
        print(thetas)
        return sum((y - numpy.dot(thetas[0:-1],X[0:2]) - thetas[-1])**2)

    gradient = Mult_gradient(ha_function, .00001, .01, 10)

    thetas = gradient.gradient(seed[0:3])

    print(thetas)

    ypredHa = numpy.dot(thetas[0:-1],XForPred[0:2]) + thetas[-1]
    plt.scatter(x,y)
    plt.plot(xForPred, ypredHa)
    plt.show()
