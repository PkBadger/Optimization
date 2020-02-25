import numpy

class Mult_gradient:
    def __init__(self, cuadratic_matrix, lineal_vector, alpha, h, treshold):
        self.cuadratic = cuadratic_matrix
        self.lineal = lineal_vector
        self.alpha = alpha
        self.h = h
        self.treshold = treshold

    def gradient(self, initial_values):
        while True:
            derivative = self.calculate_derivative(initial_values)
            initial_values = initial_values - self.alpha * derivative
            if(numpy.max(derivative) < self.treshold): break
        return initial_values
    
    def calculate_derivative(self, initial_values):
        fx = self.function(initial_values)
        diagonal = numpy.diag(numpy.full(initial_values.shape[0], self.h))

        inputs = initial_values.T + diagonal

        return numpy.matrix([[self.derivative(fx, inp.T)] for inp in inputs])

    def derivative(self, fx, fxh):
        return (self.function(fxh) - fx)/self.h

    def function(self, variable_vector):
        cuadratic = variable_vector.T.dot(self.cuadratic).dot(variable_vector) * 0.5

        lineal = self.lineal.T.dot(variable_vector)

        return numpy.sum(cuadratic + lineal)

if __name__ == "__main__":

    initial_values = numpy.matrix([[5],[1]])
    cuadratic = numpy.matrix([[2,1],[1,20]])
    lineal =   numpy.matrix([[-5],[-3]])
    
    gradient = Mult_gradient(cuadratic, lineal, .01, .01, .00000001)
    
    print(gradient.gradient(initial_values))


