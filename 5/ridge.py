import numpy
from matplotlib import pyplot as plt
from test_gradient import Mult_gradient

numpy.random.seed(100)

couter = 0
result = 1000000

seed = [1,.1,.01,.001,.0001,.00001,.000001]

m = 5
b = 2

std = 3

x = numpy.arange(0,11,1)
X = [x, x**2, x**3, x**4, x**5, x**6]

xForPred = numpy.arange(0,10.1,.1)
XForPred = [xForPred, xForPred**2, xForPred**3, xForPred**4, xForPred**5, xForPred**6]


epsilon = numpy.random.normal(0, std, (len(x))) 

y = m * x + b + epsilon
lambd = 10
def hc_function(thetas):
    global couter
    global result
    couter = couter + 1
    result = sum((y - numpy.dot(thetas[0:-1],X) - thetas[-1])**2) + lambd*(sum(thetas**2))
    if(couter == 1): print(result)
    if(couter % 80000 == 0): 
        print(result)
        print(couter)
    #print(sum((y - numpy.dot(thetas[0:-1],X) - thetas[-1])**2))
    return sum((y - numpy.dot(thetas[0:-1],X) - thetas[-1])**2) + lambd*(sum(thetas**2))

#gradient = Mult_gradient(hc_function, .00000000000036455, .0001, .001)
#alphas = numpy.linspace(.00000000000036455 ,.000000000001, 1)
alphas = [.0000000000004 ]
#alphas = numpy.arange(.000000000000001 ,.000000000001, .000000000000005)
upper_alpha = .1
lower_alpha =  .000000000000455563
#upper_alpha = .45
#lower_alpha =  .1

def calculate_alpha(upper, lower):
    delta = upper - lower
    return lower + delta/2

while True:
    couter = 0
    temp_alpha = calculate_alpha(upper_alpha, lower_alpha)
    #print("ALPHA")
    #print(temp_alpha) 
    #print("RESULT")
    #print(result) 
    gradient = Mult_gradient(hc_function, lower_alpha, .00000000001, 10)

    thetas = gradient.gradient(seed[0:7])

    if(couter > 80000):
        print("LOWER")
        lower_alpha = temp_alpha
    else:
        #print("UPPER")
        upper_alpha = temp_alpha

    if(result < 10000):
        print("FINAL ALPHA")
        print(temp_alpha) 

    #print(thetas)

    ypredHb = numpy.dot(thetas[0:-1],XForPred) + thetas[-1]
    plt.scatter(x,y)
    plt.plot(xForPred, ypredHb)
    plt.savefig('graph_images/' + str(couter) + '.png')
    plt.show()
    plt.close()
    print()
    break

