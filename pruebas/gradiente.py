import torch
import fastai
from fastai.vision.all import *
import numpy as np
import matplotlib.pyplot as plt

loss=nn.MSELoss() # Definition of least square error function
def f(a,b,c,x): return a*x**2+b*x+c # Cuadratic function
def noise(y): # Add some noise to a given function
  result=[]
  for x in y:
    result.append(y*random.uniform(0.90,1.10)+random.uniform(-0.5,0.5))
  return result 

x=torch.linspace(0,2,steps=100) # Creates spacial vector of 100 components
abc=torch.tensor([1.5,1.5,1.5],requires_grad=True) #Parameters for estimate function
estimate=f(abc[0],abc[1],abc[2],x) # Cuadratic estimate function
real =f(3,2,1,x) # Cuadratic real function
#plt.figure(1)
#plt.plot(x, real, label='real', color='blue', linestyle='-')   
#plt.plot(x, estimate, label='estimate', color='red', linestyle='--')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.grid(True)
#plt.show()
for i in range(0,20,1): #Gradient descent (20 iterations)
    estimate = f(abc[0], abc[1], abc[2], x)
    error=loss(estimate,real) # Calculate MSE 
    print("Error: ",error)
    error.backward() # Calculates the gradient of the function for each parameter with requires_grad=True and gets added as attribute to abc. 
    print("Gradient for cost function: ",abc.grad)
    with torch.no_grad(): 
      abc -=0.01*abc.grad # Moves in negative direction of the gradient to fit parameters of the function
      abc.grad.zero_() # Avoids gradient getting accumulated
abc_opt=abc # Optimal parameters
estimate_opt=f(abc_opt[0],abc_opt[1],abc_opt[2],x)
plt.figure(1)
plt.title('Before gradient descent')
plt.plot(x.numpy(), real.squeeze().numpy(), label='real', color='blue', linestyle='-')   
plt.plot(x.numpy(), estimate.squeeze().numpy(), label='estimate', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
plt.figure(2)
plt.title('After gradient descent')
plt.plot(x.numpy(), real.suqeeze().numpy(), label='real', color='blue', linestyle='-')   
plt.plot(x.numpy(), estimate_opt.squeeze().numpy(), label='estimate', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()