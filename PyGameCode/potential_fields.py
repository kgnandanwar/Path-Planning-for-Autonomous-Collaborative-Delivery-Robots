#!/usr/bin/env python
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from scipy.ndimage.morphology import distance_transform_edt as bwdist


# Potential Fields functions

def grid_map(obstacles, nrows=500, ncols=500):
    grid = np.zeros((nrows, ncols))
    # rectangular obstacles
    for obstacle in obstacles:
        x1 = meters2grid(obstacle[0][1]); x2 = meters2grid(obstacle[2][1])
        y1 = meters2grid(obstacle[0][0]); y2 = meters2grid(obstacle[2][0])
        grid[x1:x2, y1:y2] = 1
    return grid

def meters2grid(pose_m, nrows=500, ncols=500):
    if np.isscalar(pose_m):
        pose_on_grid = int( pose_m*100 + ncols/2 )
    else:
        pose_on_grid = np.array( np.array(pose_m)*100 + np.array([ncols/2, nrows/2]), dtype=int )
    return pose_on_grid
def grid2meters(pose_grid, nrows=500, ncols=500):
    if np.isscalar(pose_grid):
        pose_meters = (pose_grid - ncols/2) / 100.0
    else:
        pose_meters = ( np.array(pose_grid) - np.array([ncols/2, nrows/2]) ) / 100.0
    return pose_meters

def combined_potential(obstacles_grid, goal, influence_radius=2, attractive_coef=1./700, repulsive_coef=200, nrows=500, ncols=500):
    goal = meters2grid(goal)
    d = bwdist(obstacles_grid==0)
    d2 = (d/100.) + 1 # Rescale and transform distances
    d0 = influence_radius
    nu = repulsive_coef
    repulsive = nu*((1./d2 - 1./d0)**2)
    repulsive [d2 > d0] = 0
    [x, y] = np.meshgrid(np.arange(ncols), np.arange(nrows))
    xi = attractive_coef
    attractive = xi * ( (x - goal[0])**2 + (y - goal[1])**2 )
    f = attractive + repulsive
    return f


def gradient_planner_next(current_point, f, params):

    [gy, gx] = np.gradient(-f)
    iy, ix = np.array( meters2grid(current_point), dtype=int )
    w = 20 # smoothing window size for gradient-velocity
    vx = np.mean(gx[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2)])
    vy = np.mean(gy[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2)])
    dt = 0.01 / np.linalg.norm([vx, vy])
    V = np.array([vx, vy])*params.drone_vel
    next_point = current_point + dt*V

    return next_point, V