import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Cube:
    def __init__( self ):
        self.size = np.array( [0.01, 0.01, 0.01] )
        self.rho = 300
        self.mass = self.rho * self.size.prod()
        volDist = np.square( self.size ).dot( 1 - np.eye( 3 ) )
        self.theta = self.mass / 12 * np.diag( volDist )
        self.thetaInv = np.linalg.inv( self.theta )
        g = 9.81; self.grav0 = np.array( [0, 0, -g] )
        sG = 0.01 * self.size.min(); self.cK = self.mass * g / sG
        dL = 0.25; self.dK = 2 * np.sqrt( self.cK * self.mass ) * dL
        self.mue = 0.35
        self.vN = 0.001
        self.vertices = self.size * ( np.array(
            [[0,0,0], [0,0,1], [0,1,0], [0,1,1],
             [1,0,0], [1,0,1], [1,1,0], [1,1,1]] ) - 0.5 )
        self.indices = np.array(
            [[0,1,3,2],[4,5,7,6],
             [0,1,5,4],[2,3,7,6],
             [0,2,6,4],[1,3,7,5]] )

def cube_on_ground_dydt( t, y, cube ):

    r0S0   = y[0:3]
    pE     = y[3:7]
    v0S0   = y[7:10]
    om0KK  = y[10:13]

    pEn = pE / np.linalg.norm( pE )
    A0K, _, L = quad_to_mat( pEn )

    f0 = np.zeros( 3 )
    m0 = np.zeros( 3 )

    for rSPK in cube.vertices:
        rSP0 = A0K.dot( rSPK )
        r0P0 = r0S0 + rSP0
        s = -r0P0[2]
        if s > 0:
            v0P0 = v0S0 + A0K.dot( np.cross( om0KK, rSPK ) )
            sp = -v0P0[2]
            ff = cube.cK * s
            fd = cube.dK * sp
            fn = ff + np.clip( fd, -ff, ff )
            vg = v0P0[:2] / max( np.linalg.norm( v0P0[:2] ), cube.vN )
            fr = -cube.mue * fn * vg
            fk0 = np.append( fr, fn )
            f0 += fk0
            m0 += np.cross( rSP0, fk0 )

    r0S0p = v0S0
    pEp = 0.5 * L.T.dot( om0KK ) + np.linalg.norm( om0KK ) * ( pEn - pE )
    v0S0p = f0 / cube.mass + cube.grav0
    om0KKp = cube.thetaInv.dot( A0K.T.dot( m0 ) - np.cross( om0KK, cube.theta.dot( om0KK ) ) )
    
    return np.concatenate( ( r0S0p, pEp, v0S0p, om0KKp ) )

def quad_to_mat( pE ):

    G = np.array( [[ -pE[1],  pE[0], -pE[3],  pE[2] ],
                   [ -pE[2],  pE[3],  pE[0], -pE[1] ],
                   [ -pE[3], -pE[2],  pE[1],  pE[0] ]] )
    
    L = np.array( [[ -pE[1],  pE[0],  pE[3], -pE[2] ],
                   [ -pE[2], -pE[3],  pE[0],  pE[1] ],
                   [ -pE[3],  pE[2], -pE[1],  pE[0] ]] )

    return G.dot( L.T ), G, L

cube = Cube()
r0S0 = np.array( [0., 0., cube.size[2]] )
pE = [1., 0., 0., 0.]
v0S0 = [0.1, 0., 1.]
om0KK = [5., 50., 30.]
y0 = np.concatenate( ( r0S0, pE, v0S0, om0KK ) )
fps = 4 * 20
ts = np.arange( 0.5 * fps ) / fps

try:
    ys = np.load( "cube_on_ground.npy", allow_pickle = True )
    assert len( ts ) == len( ys.T )
    assert np.all( ys.T[0] == y0 )
except Exception as ex:
    ys = solve_ivp( cube_on_ground_dydt, ( 0, ts.max() ), y0, args = [cube], method = 'RK23', t_eval = ts ).y
    np.save( "cube_on_ground.npy", ys )

# plot path:
fg = plt.figure( figsize = ( 8, 6 ), facecolor = 'gray' )
fg.subplots_adjust( 0, 0, 1, 1 )
ax = fg.gca( projection = '3d', facecolor = 'gray' )
ax.plot3D( ys[0], ys[1], ys[2], 'gray' )

# fix aspect ratio:
limits = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
center = np.mean( limits, axis = 1 )
radius = ( np.ptp( limits, axis = 1 ) + cube.size ).max() / 2
center[2] = max( center[2], radius )
ax.auto_scale_xyz( *( center[:,np.newaxis] + [-radius, radius] ) )

# animate cube:
collection = Poly3DCollection( cube.vertices[cube.indices], edgecolor = "k" )
collection.set_facecolor( ( "r", "r", "g", "g", "b", "b" ) )
ax.add_collection3d( collection )

def update( y ):
    verts = y[0:3] + cube.vertices.dot( quad_to_mat( y[3:7] )[0].T )
    collection.set_verts( verts[cube.indices] )

animation = FuncAnimation( fg, update, frames = ys.T, interval = 50 )

plt.show()
