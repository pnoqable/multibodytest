# from numba import njit
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
y0 = np.concatenate( ( r0S0, pE, v0S0, om0KK ) )

# solve ODE
ts = np.linspace( 0, 1.5, 151 )
sol = solve_ivp( cuboid_dydt, ( 0, 1.5 ), y0, t_eval = ts, args = ( Theta, ThetaInv, grav0 ), dense_output = True )

vertices = np.array( [[0,0,0],[0,0,c],[0,b,0],[0,b,c],
                      [a,0,0],[a,0,c],[a,b,0],[a,b,c]], dtype = np.float_ )
vertices -= np.mean( vertices, axis = 0 )
indices = np.array( [[0,1,3,2],[4,5,7,6],
                     [0,1,5,4],[2,3,7,6],
                     [0,2,6,4],[1,3,7,5]], dtype = np.int_ )

collection = Poly3DCollection( vertices[indices], edgecolor = "k" )

figure = plt.figure()
ax = plt.axes( projection = '3d' )
ax.set_xlim([-0.2, 0.2])
ax.set_ylim([-0.2, 0.2])
ax.set_zlim([-0.2, 0.2])
ax.add_collection3d( collection )

def update( y ):
    collection.set_verts( vertices.dot( quad_to_mat( y[3:7] )[0].T )[indices] )

animation = FuncAnimation( figure, update, frames = sol.y.T, interval = 1 )

plt.show()
