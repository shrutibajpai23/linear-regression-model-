#task 1 : implement mathematical formula for linear regression 

import numpy as np

#generate synthetic data
np.random.seed(42)
X=2*np.random.rand(100,1)
Y=4+3*X+np.random.randn(100,1)

#add bias term to feature matrix 
X_b=np.c_[np.ones((100,1)),X]

#initialise parameters 
theta=np.random.randn(2,1)
learning_rate=0.1
iteration=1000



def predict(X,theta):
    return np.dot(X,theta)

#task 2: use gradient descent to optimize model parameters 
def gradient_descent(X,Y,theta,learning_rate,iteration):
    m = len(Y)
    for i in range(iteration):
        gradients = 1/m * X.T.dot(X.dot(theta) - Y)
        theta -= learning_rate * gradients
    return theta 
    

#task 3 : calculate evaluation matrix 

def mean_squared_error(Y_true,Y_pred):
    return np.mean((Y_true-Y_pred)**2)

def r_squared(Y_true,Y_pred):
    ss_res=np.sum((Y_true-Y_pred)**2)
    ss_tot=np.sum((Y_true-np.mean(Y_true))**2)
    return 1-(ss_res/ss_tot)


#perform gradient descent 
theta_optimized = gradient_descent(X_b, Y, theta, learning_rate, iteration)

#predicitons and evaluations 
Y_pred=predict(X_b,theta_optimized)
mse=mean_squared_error(Y,Y_pred)
r2=r_squared(Y,Y_pred)

print("optimized parameters(theta)" , theta_optimized)
print("Mean Squared Error: ",mse)
print("R squared: ",r2)