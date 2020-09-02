# from numba import njit
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# function that returns dy/dt
# @njit( cache = True )
def cuboid_dydt( t, y, Theta, ThetaInv, grav0 ):

    r0S0  = y[0:3]
    pE    = y[3:7]
    v0S0  = y[7:10]
    om0KK = y[10:13]
    
    pEn = pE / np.linalg.norm( pE ); A0K, G, L = quad_to_mat( pEn )

    r0S0p = v0S0;  pEp    = 0.5 * np.dot( L.T, om0KK ) + np.linalg.norm( om0KK ) * ( pEn - pE )
    v0S0p = grav0; om0KKp = np.dot( ThetaInv, -np.cross( om0KK, np.dot( Theta, om0KK ) ) )

    return np.concatenate( ( r0S0p, pEp, v0S0p, om0KKp ) )

# @njit( cache = True )
def quad_to_mat( pE ):

    G = np.array( [[ -pE[1],  pE[0], -pE[3],  pE[2] ],
                   [ -pE[2],  pE[3],  pE[0], -pE[1] ],
                   [ -pE[3], -pE[2],  pE[1],  pE[0] ]] )
    
    L = np.array( [[ -pE[1],  pE[0],  pE[3], -pE[2] ],
                   [ -pE[2], -pE[3],  pE[0],  pE[1] ],
                   [ -pE[3],  pE[2], -pE[1],  pE[0] ]])

    return np.dot( G, L.T ), G, L

a = 0.1; b = 0.05; c = 0.01
rho   = 700
grav0 = np.array( [ 0, 0, -9.81 ] )
mass  = rho * a * b * c
Theta = mass / 12 * np.diag( [ b**2+c**2, c**2+a**2, a**2+b**2 ] )
ThetaInv = np.linalg.inv( Theta )

# initial condition
r0S0  = [ 0, 0, 0 ]
pE    = [ 1, 0, 0, 0 ]
v0S0  = [ 0, 0, 7 ]
om0KK = [ 0, 25, 0 ]
om0KK = np.array( om0KK ) + np.max( om0KK ) / 100
x0 = np.concatenate( ( r0S0, pE, v0S0, om0KK ) )

# solve ODE
sol = solve_ivp( cuboid_dydt, ( 0, 1.5 ), x0, args = ( Theta, ThetaInv, grav0 ), dense_output = True )
denseT = np.linspace( 0, 1.5, 1500 )
denseY = sol.sol( denseT )

# plot results
plt.plot( denseT, denseY[10:13].T, color = 'gray' )
plt.plot( sol.t, sol.y[10], 'o', sol.t, sol.y[11], '^', sol.t, sol.y[12], '*' )
plt.xlabel( '[s]' )
plt.ylabel( '[RAD/s]' )
plt.show()

# plot 3d
ax = plt.axes( projection = '3d' )
max = np.abs( sol.y[10:13] ).max()
ax.set_xlim([-max, max])
ax.set_ylim([-max, max])
ax.set_zlim([-max, max])
ax.plot3D( denseY[10], denseY[11], denseY[12], 'gray' )
ax.scatter3D( sol.y[10], sol.y[11], sol.y[12] );
plt.show()

# solve ODE
sol = solve_ivp( cuboid_dydt, ( 0, 30 ), x0, args = ( Theta, ThetaInv, grav0 ) )

# plot results
pEpE = np.sum( sol.y[3:7] * sol.y[3:7], axis = 0 )
plt.plot( sol.t, 100 * pEpE - 100 )
plt.xlabel('[s]')
plt.ylabel('[%]')
plt.show()
