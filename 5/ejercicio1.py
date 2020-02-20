from matplotlib import pyplot
from main import Gradient_aproximation

if __name__ == "__main__":
    gradient = Gradient_aproximation(lambda x: x**4 + 7*x**3 + 5*x**2 -17*x +3)

    gradient.calculate_gradient(0, .01, .01)

    print(gradient.results[-1])

    gradient.graph()
    gradient.graph_original(2)

    pyplot.show()

    