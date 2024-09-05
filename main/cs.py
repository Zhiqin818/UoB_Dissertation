# -*- coding: utf-8 -*-
'''
This code is implemented by Chauby, it is free for everyone.
Email: chaubyZou@163.com
'''

#%% import package
import numpy as np

#%% define canonical system
class CanonicalSystem():
    def __init__(self, alpha_x=1.0, dt=0.01, type='discrete'):
        self.x = 1.0
        self.alpha_x = alpha_x
        self.dt = dt
        self.dmp_type = type

        if type == 'discrete':
            self.run_time = 1.0
        elif type == 'rhythmic':
            self.run_time = 2*np.pi
        else:
            print('Initialize Canonical system failed, can not recognize DMP type: ' + type)
        
        self.timesteps = round(self.run_time / self.dt)

        self.reset_state()

    def run(self, **kwargs): # run to goal state
        if 'tau' in kwargs:
            timesteps = int(self.timesteps / kwargs['tau'])
        else:
            timesteps = self.timesteps

        self.reset_state()
        self.x_track = np.zeros(timesteps)

        if self.dmp_type == 'discrete':
            for t in range(timesteps):
                self.x_track[t] = self.x
                self.step_discrete(**kwargs)
        elif self.dmp_type == 'rhythmic':
            for t in range(timesteps):
                self.x_track[t] = self.x
                self.step_rhythmic(**kwargs)

        return self.x_track
    
    def reset_state(self): # reset state
        self.x = 1.0

    def step_discrete(self, tau=1.0):
        dx = -self.alpha_x*self.x*self.dt
        self.x += tau*dx
        return self.x


    def step_rhythmic(self, tau=1.0):
        self.x += tau*self.dt
        return self.x
