import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        
        self.closeCount = 0

        # Goal
        #Reviewer said to focus on Z axis
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self, done):
        """Uses current pose of sim to return reward."""
        # penalize crash
        if (done and (self.sim.time < self.sim.runtime)): 
            #print("crashed")
            reward = -1.5

        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #print("v2", self.sim.v[2])
        #reward = 1.-.003*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #reward = (1 - abs(self.sim.pose[:3] - self.target_pos).sum()) + self.sim.v[2]
        #print(self.sim.v[2], self.target_pose[2])
        
        #reward = 1- np.tanh(abs(self.sim.pose[:1] - self.target_pos[2]).sum())

        reward = 1- .003*(abs(self.sim.pose[:1] - self.target_pos[2]).sum())
        
        
        mypose = self.sim.pose[:1][0]
        tpose =  self.target_pos[2]
        #print(mypose, tpose)

        #encourage moving up if too low, dont want to move up if too high
 
        if (mypose < tpose):
            reward += np.clip(self.sim.v[2], -.1, .1)
        elif (abs(mypose - tpose) > tpose): #more than twice as high! too high!
            print("too high")
            reward -= np.clip(self.sim.v[2], -.1, .1)
                          
        reward = np.tanh(reward) #normalize
                          
        #encourage being close to the target
        closeto = 1
        if abs(self.sim.pose[:1] - self.target_pos[2].sum()) < closeto:  
        # if the agent close to target by some specified amount, significantly increase reward
            reward += 1.5 
            print("sort of close", reward)
            if abs(self.sim.pose[:1] - self.target_pos[2].sum()) < (closeto/2):
                print("so close! current Z: ", self.sim.pose[:1])
                reward += 2
                self.closeCount += 1
                print("very close", reward)
        
        #print("reward", reward)
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(done) 
            pose_all.append(self.sim.pose)
        #print("pose_all", pose_all)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.closeCount = 0
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state