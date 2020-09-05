import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

vertices = np.array( [[0,0,0],[0,0,1],[0,1,0],[0,1,1],
                      [1,0,0],[1,0,1],[1,1,0],[1,1,1]], dtype = np.float_ ) * 2 - 1
indices = np.array( [[0,1,3,2],[4,5,7,6],
                     [0,1,5,4],[2,3,7,6],
                     [0,2,6,4],[1,3,7,5]], dtype = np.int_ )

collection = Poly3DCollection( vertices[indices], edgecolor="k" )

figure = plt.figure()
axes = plt.axes( projection = '3d' )
axes.set_xlim([-1, 1])
axes.set_ylim([-1, 1])
axes.set_zlim([-1, 1])
axes.add_collection3d( collection )

def rotation( alpha ):
    cosa, sina = np.cos( alpha ), np.sin( alpha )
    return np.array( [[  cosa, -sina, 0 ],
                      [  sina,  cosa, 0 ],
                      [     0,     0, 1 ]],
                     dtype = np.float_ )

def update( alpha ):
    collection.set_verts( vertices.dot( rotation( alpha ).T )[indices] )

alphas = np.arange( 12 ) / 24 * np.pi
animation = FuncAnimation( figure, update, frames = alphas, interval = 1 )

plt.show()
