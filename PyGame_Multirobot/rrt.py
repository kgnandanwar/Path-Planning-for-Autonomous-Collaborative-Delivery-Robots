#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from math import *
from random import random
from scipy.spatial import ConvexHull
from matplotlib import path
import time


def isCollisionFreeVertex(obstacles, xy):
    collFree = True

    for obstacle in obstacles:
        hull = path.Path(obstacle)
        collFree = not hull.contains_points([xy])
        if hull.contains_points([xy]):
            return collFree

    return collFree


def isCollisionFreeEdge(obstacles, closest_vert, xy):
    closest_vert = np.array(closest_vert); xy = np.array(xy)
    collFree = True
    l = norm(closest_vert - xy)
    map_resolution = 0.01; M = int(l / map_resolution)
    if M <= 2: M = 20
    t = np.linspace(0,1,M)
    for i in range(1,M-1):
        p = (1-t[i])*closest_vert + t[i]*xy # calculate configuration
        collFree = isCollisionFreeVertex(obstacles, p) 
        if collFree == False: return False

    return collFree

class Node:
    def __init__(self):
        self.p     = [0, 0]
        self.i     = 0
        self.iPrev = 0


def closestNode(rrt, p):
    distance = []
    for node in rrt:
        distance.append( sqrt((p[0] - node.p[0])**2 + (p[1] - node.p[1])**2) )
    distance = np.array(distance)
    
    dmin = min(distance)
    ind_min = distance.tolist().index(dmin)
    closest_node = rrt[ind_min]

    return closest_node


def rrt_path(obstacles, xy_start, xy_goal, params):
    rrt = []
    start_node = Node()
    start_node.p = xy_start
    start_node.i = 0
    start_node.iPrev = 0
    rrt.append(start_node)
    nearGoal = False 
    minDistGoal = params.minDistGoal 
    d = params.extension 

    start_time = time.time()
    iters = 0
    print('Configuration space sampling started ...')
    while not nearGoal: # and iters < maxiters:
        # Sample point
        rnd = random()
        if rnd < params.goal_prob:
            xy = xy_goal
        else:
            
            xy = np.array([random()*2*params.world_bounds_x[1]-params.world_bounds_x[1], random()*2*params.world_bounds_x[1]-params.world_bounds_x[1]]) # Should be a 2 x 1 vector
        collFree = isCollisionFreeVertex(obstacles, xy)
        if not collFree:
            iters += 1
            continue

        closest_node = closestNode(rrt, xy)
        new_node = Node()
        new_node.p = closest_node.p + d * (xy - closest_node.p)
        new_node.i = len(rrt)
        new_node.iPrev = closest_node.i

        collFree = isCollisionFreeEdge(obstacles, closest_node.p, new_node.p)
        if not collFree:
            iters += 1
            continue
        
        if params.animate:
            plt.plot(new_node.p[0], new_node.p[1], 'bo',color = 'blue', markersize=5) # VERTICES
            plt.plot([closest_node.p[0], new_node.p[0]], [closest_node.p[1], new_node.p[1]], color='blue') # EDGES
            plt.draw()
            plt.pause(0.01)


        rrt.append(new_node)
        if norm(np.array(xy_goal) - np.array(new_node.p)) < minDistGoal:
            # Add last, goal node
            goal_node = Node()
            goal_node.p = xy_goal
            goal_node.i = len(rrt)
            goal_node.iPrev = new_node.i
            if isCollisionFreeEdge(obstacles, new_node.p, goal_node.p):
                rrt.append(goal_node)
                P = [goal_node.p]
            else: P = []

            end_time = time.time()
            nearGoal = True
            print('RRT is constructed after %.2f seconds:' % (end_time - start_time))

        iters += 1

    print('Retriving the path from RRT...')
    i = len(rrt) - 1
    while True:
        i = rrt[i].iPrev
        P.append(rrt[i].p)
        if i == 0:
            break
    P = np.array(P)

    return P
