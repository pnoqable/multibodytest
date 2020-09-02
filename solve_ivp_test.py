import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# function that returns dy/dt
def model(t, y):
    k = 0.3
    dydt = -k * y
    return dydt

# initial condition
y0 = 5

# time points
t = np.linspace(0,20,21)

# solve ODE
y = odeint( model, y0, t, tfirst=True )

# plot results
plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')

plt.show()

sol = solve_ivp( model, [0,20], [y0], t_eval=t, dense_output = True )

denseT = np.linspace(0,20,201)
plt.plot( denseT, sol.sol(denseT)[0] )
plt.xlabel('time')
plt.ylabel('res.y(t)')

plt.show()
