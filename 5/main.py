from matplotlib import pyplot
import numpy

class Gradient_aproximation:
    def __init__(self, function):
        self.function = function
        self.results = []
        

    def calculate_gradient(self, x0, alpha, h):
        self.results.append(x0)
        for _ in range(10):
            x0 = x0 - alpha * self.calculate_derivative(x0, h)
            self.results.append(x0)

        return self.results

    def calculate_derivative(self, x0, h):
        return (self.function(x0 + h) - self.function(x0)) / h

    def graph(self):
        y = []
        for result in self.results:
            y.append(self.function(result))
        pyplot.scatter(self.results, y)

    def graph_original(self, limit=3):
        y = []
        x = numpy.arange(min(self.results) - limit,max(self.results) + limit,.1)
        for x0  in x:
            y.append(self.function(x0))
        pyplot.plot(x,y)


if __name__ == "__main__":
    gradient = Gradient_aproximation(lambda x: x**2)

    gradient.calculate_gradient(10, .1, .1)

    print(gradient.results)

    gradient.graph()
    gradient.graph_original()

    pyplot.show()
