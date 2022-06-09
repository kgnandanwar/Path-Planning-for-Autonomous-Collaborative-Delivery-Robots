# Path-Planning-for-Autonomous-Collaborative-Delivery-Robots

This project revolves around the idea of implementing a Motion Planning Algorithm on a group of Autonomous Mobile Robots for multiple purposes, say food delivery, package delivery in a warehouse, etc. The aim is to focus and delve deeper into this domain by exploring RRT & RRT* algorithm and implementing it on autonomous swarm robots with evasive movement capabilities. 

We began the project with an environment setup. Since there was a bit of uncertainty in the potential of the various environments to launch multiple robots, we began implementing the basic environment set up on four different tools: PyGame, V-REP, Gazebo and CARLA. After weighing the pros/cons(say community support) of different software and the errors faced, we moved ahead with PyGame and Gazebo. 

We started with the implementation of the Motion Planning algorithm, first on a single robot and then on a multi-robot system.
![pygame_simulation](https://user-images.githubusercontent.com/76609547/172793169-369d5678-b22d-481c-aebe-b06d3df10a2e.gif)

Kindly refer to the above single robot simulation(in PyGame), below multi-robot simulation(in Rviz) and the files for more details.
![simulation](https://user-images.githubusercontent.com/76609547/172793156-a79dd257-c5c2-406a-be22-927bcb6b7556.gif)



