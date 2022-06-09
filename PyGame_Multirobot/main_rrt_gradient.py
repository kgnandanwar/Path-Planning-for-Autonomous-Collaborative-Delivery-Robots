#!/usr/bin/env python

import pygame
import numpy as np
import matplotlib.pyplot as plt

from tools import *
from rrt import *
from potential_fields import *


def move_obstacles(obstacles, params):

    obstacles[-3] += np.array([0.02, 0.0]) * params.drone_vel
    obstacles[-2] += np.array([-0.005, 0.005]) * params.drone_vel
    obstacles[-1] += np.array([0.0, 0.01]) * params.drone_vel
    return obstacles

class Params:
    def __init__(self):
        self.animate = 1 # show RRT construction, set 0 to reduce time of the RRT algorithm
        self.visualize = 1 # show constructed paths at the end of the RRT and path smoothing algorithms
        self.maxiters = 5000 # max number of samples to build the RRT
        self.goal_prob = 0.05 # with probability goal_prob, sample the goal
        self.minDistGoal = 0.25 # [m], min distance os samples from goal to add goal node to the RRT
        self.extension = 0.4 # [m], extension parameter: this controls how far the RRT extends in each step.
        self.world_bounds_x = [-2.5, 2.5] # [m], map size in X-direction
        self.world_bounds_y = [-2.5, 2.5] # [m], map size in Y-direction
        self.drone_vel = 4.0 # [m/s]
        self.ViconRate = 100 # [Hz]
        self.max_sp_dist = 0.3 * self.drone_vel # [m], maximum distance between current robot's pose and the sp from global planner
        self.influence_radius = 1.22 # potential fields radius, defining repulsive area size near the obstacle
        self.goal_tolerance = 0.05 # [m], maximum distance threshold to reach the goal
        self.num_robots = 1
        self.moving_obstacles = 3 # move small cubic obstacles or not


class Robot:
    def __init__(self):
        self.sp = [0, 0]
        self.sp_global = [0,0]
        self.route = np.array([self.sp])
        self.f = 0
        self.leader = False
        self.vel_array = []

    def local_planner(self, obstacles, params):
        obstacles_grid = grid_map(obstacles)
        self.f = combined_potential(obstacles_grid, self.sp_global, params.influence_radius)
        self.sp, self.vel = gradient_planner_next(self.sp, self.f, params)
        self.vel_array.append(norm(self.vel))
        self.route = np.vstack( [self.route, self.sp] )


class Robot2:
    def __init__(self):
        self.sp = [0, 0]
        self.sp_global = [0,0]
        self.route = np.array([self.sp])
        self.f = 0
        self.leader = False
        self.vel_array = []

    def local_planner(self, obstacles, params):
        obstacles_grid = grid_map(obstacles)
        self.f = combined_potential(obstacles_grid, self.sp_global, params.influence_radius)
        self.sp, self.vel = gradient_planner_next(self.sp, self.f, params)
        self.vel_array.append(norm(self.vel))
        self.route = np.vstack( [self.route, self.sp] )


# Initialization
params = Params()
xy_start = np.array([1.4, 0.9])
xy_goal =  np.array([1.5, -1.4])

xy_start2 = np.array([1.4,2.0])
xy_goal2 = np.array([-2.0,-1.4])


""" Obstacles map construction """
obstacles = [
              # bugtrap
              np.array([[0.5, 0], [2.5, 0.], [2.5, 0.3], [0.5, 0.3]]),
              np.array([[0.5, 0.3], [0.8, 0.3], [0.8, 1.5], [0.5, 1.5]]),
              np.array([[0.5, 1.5], [1.5, 1.5], [1.5, 1.8], [0.5, 1.8]]),

              # moving obstacle
              np.array([[-2.3, 2.0], [-2.2, 2.0], [-2.2, 2.1], [-2.3, 2.1]]),
              np.array([[2.3, -2.3], [2.4, -2.3], [2.4, -2.2], [2.3, -2.2]]),
              np.array([[-1.5, -2.3], [-1.4, -2.3], [-1.4, -2.2], [-1.5, -2.2]]),
            ]

passage_width = 0.25
passage_location = 0.0

robots = []
for i in range(params.num_robots):
    robots.append(Robot())
robot1 = robots[0]; robot1.leader=True

robots2 = []
for i in range(params.num_robots):
    robots2.append(Robot2())
robot21 = robots2[0]; robot21.leader=True

# postprocessing variables:
mean_dists_array = []
max_dists_array = []

mean_dists_array2 = []
max_dists_array2 = []
# Layered Motion Planning: RRT (global) + Potential Field (local)
if __name__ == '__main__':
    plt.figure(figsize=(10,10))
    draw_map(obstacles)
    plt.plot(xy_start[0],xy_start[1],'bo',color='red', markersize=20, label='start')
    plt.plot(xy_goal[0], xy_goal[1],'bo',color='green', markersize=20, label='goal')

    plt.plot(xy_start2[0],xy_start2[1],'bo',color='red', markersize=20, label='start2')
    plt.plot(xy_goal2[0], xy_goal2[1],'bo',color='green', markersize=20, label='goal2')

    P = rrt_path(obstacles, xy_start, xy_goal, params)

    traj_global = waypts2setpts(P, params); P = np.vstack([P, xy_start])
    plt.plot(P[:,0], P[:,1], linewidth=3, color='orange', label='Global planner path')
    plt.pause(0.1)

    sp_ind = 0
    robot1.route = np.array([traj_global[0,:]])
    robot1.sp = robot1.route[-1,:]

    P2 = rrt_path(obstacles, xy_start2, xy_goal2, params)

    traj_global2 = waypts2setpts(P2, params); P2 = np.vstack([P2, xy_start2])
    plt.plot(P2[:,0], P2[:,1], linewidth=3, color='orange', label='Global planner path2')
    plt.pause(0.1)

    sp_ind2 = 0
    robot21.route = np.array([traj_global2[0,:]])
    robot21.sp = robot21.route[-1,:]

    while True: # loop through all the setpoint from global planner trajectory, traj_global
        dist_to_goal = norm(robot1.sp - xy_goal)
        dist_to_goal2 = norm(robot21.sp - xy_goal2)
        if dist_to_goal < params.goal_tolerance and dist_to_goal2 < params.goal_tolerance: # [m]
            print('Goal is reached')
            break
        

        if params.moving_obstacles: obstacles = move_obstacles(obstacles, params) # change poses of some obstacles on the map

        # leader's setpoint from global planner
        robot1.sp_global = traj_global[sp_ind,:]
        # correct leader's pose with local planner
        robot1.local_planner(obstacles, params)

        # centroid pose:
        centroid = 0
        for robot in robots:
        	centroid += robot.sp / len(robots)
        # dists to robots from the centroid:
        dists = []
        for robot in robots:
        	dists.append( norm(centroid-robot.sp) )
        mean_dists_array.append(np.mean(dists))
        max_dists_array.append(np.max(dists))


        #ROBOT2


        # leader's setpoint from global planner
        robot21.sp_global = traj_global2[sp_ind2,:]
        # correct leader's pose with local planner
        robot21.local_planner(obstacles, params)

        # centroid pose:
        centroid2 = 0
        for robot in robots2:
        	centroid2 += robot.sp / len(robots2)
        # dists to robots from the centroid:
        dists2 = []
        for robot in robots2:
        	dists2.append( norm(centroid2-robot.sp) )
        mean_dists_array2.append(np.mean(dists2))
        max_dists_array2.append(np.max(dists2))

        # vizualization
        plt.cla()
        plt.plot(centroid[0], centroid[1], '*', color='blue', markersize=7)
        plt.plot(centroid2[0], centroid2[1], '*', color='blue', markersize=7)

        draw_map(obstacles)
        if params.num_robots == 1:
            draw_gradient(robots[0].f)
        else:
            draw_gradient(robots[1].f)

        if params.num_robots == 1:
            draw_gradient(robots2[0].f)
        else:
            draw_gradient(robots2[1].f)

        for robot in robots[1:]: plt.plot(robot.sp[0], robot.sp[1], '^', color='blue', markersize=10, zorder=15) # robots poses
        plt.plot(robot1.sp[0], robot1.sp[1], '^', color='green', markersize=10, zorder=15) # robots poses
        plt.plot(robot1.route[:,0], robot1.route[:,1], linewidth=2, color='green', zorder=10)
        plt.plot(P[:,0], P[:,1], linewidth=3, color='orange')
        plt.plot(xy_start[0],xy_start[1],'bo',color='red', markersize=20)
        plt.plot(xy_goal[0], xy_goal[1],'bo',color='green', markersize=20)
        
        for robot in robots2[1:]: plt.plot(robot.sp[0], robot.sp[1], '^', color='blue', markersize=10, zorder=15) # robots poses
        plt.plot(robot21.sp[0], robot21.sp[1], '^', color='green', markersize=10, zorder=15) # robots poses
        plt.plot(robot21.route[:,0], robot21.route[:,1], linewidth=2, color='green', zorder=10)
        plt.plot(P2[:,0], P2[:,1], linewidth=3, color='orange')
        plt.plot(xy_start2[0],xy_start2[1],'bo',color='red', markersize=20)
        plt.plot(xy_goal2[0], xy_goal2[1],'bo',color='green', markersize=20)
        plt.legend()
        plt.draw()
        plt.pause(0.01)

        # update loop variable
        if sp_ind < traj_global.shape[0]-1 and norm(robot1.sp_global - robot1.sp) < params.max_sp_dist: sp_ind += 1
        # update loop variable
        if sp_ind2 < traj_global2.shape[0]-1 and norm(robot21.sp_global - robot21.sp) < params.max_sp_dist: sp_ind2 += 1


# close windows if Enter-button is pressed
plt.draw()
plt.pause(0.1)
input('Hit Enter to close')
plt.close('all')