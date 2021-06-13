'''
Mimo_RL_simple environment for ICC Tutorial - June 14, 2021
Tutorial 14: Machine Learning for MIMO Systems with Large Arrays
Aldebaro Klautau (UFPA), Nuria Gonzalez-Prelcic (NCSU) and Robert W. Heath Jr. (NCSU)
From: https://github.com/lasseufpa/ml4comm-icc21/
'''
import numpy as np
from numpy.random import randint
from bidict import bidict
import itertools
import gym
from gym import spaces

class Mimo_RL_simple(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Mimo_RL_simple, self).__init__()
        self.__version__ = "0.1.0"

        #TODO need to conclude this code for option "False". Use "True"
        #Use True to represent the state by an integer index, as in a 
        #finite Markov decision process
        self.observation_equals_state_index = True

        self.episode_duration = 30 #number of steps in an episode
        self.penalty = 100 #if not switching among users
        #a single user can be served by the base station
        self.Nu = 2 #number of users
        self.Nb = 64 #number of beams
        self.Na = 3 #user must be allocated at least once in Na allocations
        self.grid_size = 6 #define grid size: grid_size x grid_size
        #directions each user takes (left, right, up, down). Chosen at the
        #beginning of each episode
        self.users_directions_indices = np.zeros(self.Nu)
        #directions: up, down, right, left
        self.position_updates = np.array([[0,1],[0,-1],[1,0],[-1,0]])

        #We adopt bidirectional maps based on https://pypi.org/project/bidict/
        self.bidict_actions = convert_list_of_possible_tuples_in_bidct(self.get_all_possible_actions())
        self.bidict_states = convert_list_of_possible_tuples_in_bidct(self.get_all_possible_states())
        #self.bidict_rewards = convert_list_of_possible_tuples_in_bidct()
        #I don't need a table for all the rewards. I will generate them as we go. 
        #Otherwise I would implement:
        #def get_all_possible_rewards(self):
        self.reward = 0

        #TODO could get this info from the above bidicts
        self.S = len(self.get_all_possible_states())        
        self.A = len(self.get_all_possible_actions())

        #Define spaces to make environment compatible with the library Stable Baselines
        self.action_space = spaces.Discrete(self.get_num_actions())
        if self.observation_equals_state_index:
            self.observation_space = spaces.Discrete(self.get_num_states())
        else:
            high_value = np.maximum(self.Nu, self.grid_size)-1
            #TODO need to conclude this code
            self.observation_space = spaces.Box(low=0, high=high_value,
                                    shape=( (2*self.Nu + (self.Na-1)),), 
                                    dtype=np.uint8)
        
        #keep current state information based only on its index
        self.current_state_index = 0

        self.currentIteration = 0 #continuous, count time and also TTIs
        self.reset() #create variables

    def step(self, action_index):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action :
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
            reward (float) :
            episode_over (bool) :
            info (dict) :
        """
        #interpret action: convert from index to useful information
        scheduled_user, beam_index = self.interpret_action(action_index)

        #get current state
        positions, previously_scheduled = self.interpret_state(self.current_state_index)

        throughput = self.get_througput(beam_index)
        self.reward = throughput

        allocated_users = np.array(previously_scheduled)
        allocated_users = np.append(allocated_users, scheduled_user)

        if len(np.unique(allocated_users)) != self.Nu:
            self.reward = throughput - self.penalty
        
        #update for next iteration
        previous_state_index = self.current_state_index
        #loop to shift to the left
        allocated_users_tuple = tuple(allocated_users[1:])        
        #get new positions for the users. Note that this does not depend
        #on the action taken by the agent
        new_positions = self.update_users_positions(positions) 
        self.current_state_index = self.convert_state_to_index(tuple(new_positions),tuple(allocated_users_tuple))

        #check if episode has finished
        gameOver = False
        if self.currentIteration == self.episode_duration:
            ob = self.reset()
            gameOver = True  # game ends
        else:
            ob = self.get_state()
  
        # history version with actions and states
        history = {"time": self.currentIteration,
                   "action_t": self.interpret_action(action_index),
                   "state": self.interpret_state(previous_state_index),
                   "positions": positions,
                   "reward": self.reward,
                   #"users_directions_indices": self.users_directions_indices,
                   "next_state": self.interpret_state(self.current_state_index)}
        
        self.currentIteration += 1 #update iteration counter
        return ob, self.reward, gameOver, history

    def get_num_states(self):
        return self.S

    def get_num_actions(self):
        return self.A

    def get_current_reward(self):
        return self.reward

    #note that bidict cannot hash numpy arrays. We will use tuples
    def get_all_possible_actions(self):
        '''Nu is the number of users and Nb the number of beam pairs'''
        all_served_users = range(self.Nu)
        list_beam_indices = range(self.Nb)
        all_actions = [(a,b) for a in all_served_users for b in list_beam_indices]
        return all_actions

    #note that bidict cannot hash numpy arrays. We will use tuples
    def get_all_possible_states(self):
        #positions: we are restricted to square M x M grids
        positions_x_axis = np.arange(self.grid_size)
        #positions_y_axis = np.arange(self.grid_size)
        all_positions_single_user = list(itertools.product(positions_x_axis, repeat=2))
        all_positions = list(itertools.product(all_positions_single_user, repeat=self.Nu))

        #previously scheduled users
        previously_scheduled = list(itertools.product(np.arange(self.Nu), repeat=self.Na-1))
        #print(previously_scheduled)
        
        all_states = list(itertools.product(all_positions, previously_scheduled))
        #all_states = [(a,b) for a in all_positions for b in previously_scheduled]
        return all_states

    def convert_state_to_index(self,positions,previously_scheduled):
        state = (positions, previously_scheduled)
        state_index = self.bidict_states.inv[state]
        return state_index

    def interpret_action(self, action_index):
        action = self.bidict_actions[action_index]
        scheduled_user = action[0]
        beam_index = action[1]
        return scheduled_user, beam_index

    def interpret_state(self, state_index):
        state = self.bidict_states[state_index]
        positions = state[0]
        previously_scheduled = state[1]
        return positions, previously_scheduled

    def update_users_positions(self, positions):
        positions_as_array = np.array(positions)
        new_positions = list()        
        for u in range(self.Nu):
            new_position_array = positions_as_array[u] + self.position_updates[self.users_directions_indices[u]]
            #wrap-around grid:
            new_position_array[np.where(new_position_array>self.grid_size-1)] = 0
            new_position_array[np.where(new_position_array<0)] = self.grid_size-1
            #new_position_array = np.remainder(new_position_array, self.grid_size)
            new_positions.append(tuple(new_position_array))
        return tuple(new_positions)

    #TODO
    def get_througput(self, beam_index):
        return beam_index

    def get_state(self):
        """Get the current observation.
        """
        if self.observation_equals_state_index:
            return self.current_state_index
        else:
            return self.bidict_states[self.current_state_index]

    def render(self, mode='human'):
        pass
    
    def close (self):
        pass

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.currentIteration = 0
        self.current_state_index = 0 #assumes first state is always 0
        self.users_directions_indices = randint(0,4,size=(self.Nu,))
        self.current_state_index = randint(0, self.get_num_states())
        return self.get_state()

    #This was needed in previous versions but it does not seem needed anymore
    # from gym.utils import seeding
    # def seed(self, seed=None):
    #    self.np_random, seed = seeding.np_random(seed)
    #    return [seed]

    def numberOfActions(self):
        return self.A

    def numberOfObservations(self):
        return self.S

def convert_list_of_possible_tuples_in_bidct(list_of_tuples):
    #assume there are no repeated elements
    N = len(list_of_tuples)
    this_bidict = bidict()
    for n in range(N):
        this_bidict.put(n,list_of_tuples[n])
    return this_bidict

def test_bidct():
    x=list()
    y=np.zeros((3,1))
    #x.append(y) #does not work because unhashable type: 'numpy.ndarray'
    x.append((3,5,'a'))
    x.append((3,4,'a'))
    x.append(('b'))
    bidict = convert_list_of_possible_tuples_in_bidct(x)
    print(bidict[1])
    print(bidict.inv['b'])

    #test return
    x = np.random.randn(3,4)
    print(x)
    a = tuple(x.flatten())
    print(a)
    print(len(a))
    b = np.array(a).reshape((3,4))
    print(b)
    exit(-1)

if __name__ == '__main__':
    #test_bidct()
    #exit(1)
    env = Mimo_RL_simple()
    print('Actions=', env.get_all_possible_actions())
    print('###################')
    print('States=', env.get_all_possible_states())
    print('Action example', env.bidict_actions[3])
    print('State example', env.bidict_states[7])

    #x = env.get_state()
    #print(x)
    #exit(-1)

    N = 61
    Na = env.get_num_actions()
    print('# actions = ', Na)
    print('# states = ', env.get_num_states())
    for i in range(N):
        action_index = int(np.random.choice(Na,1)[0])
        ob, reward, gameOver, history = env.step(action_index)
        if gameOver:
            print('Game over! End of episode.')
            #env = EnvironmentMassiveMIMO()
        print(history)

from stable_baselines.common.env_checker import check_env
check_env(env)
