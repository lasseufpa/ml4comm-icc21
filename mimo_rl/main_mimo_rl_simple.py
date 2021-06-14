from numpy.random import randint
from beamforming_calculation import AnalogBeamformer
from render_mimo_rl_simple import Mimo_RL_render
from env_mimo_rl_simple import Mimo_RL_Simple_Env
from channel_mimo_rl_simple import Grid_Mimo_Channel

if __name__ == '__main__':
    num_antenna_elements=32
    grid_size=6
    mimo_RL_Environment = Mimo_RL_Simple_Env(num_antenna_elements=num_antenna_elements, grid_size=grid_size)

    num_steps = 500
    num_actions = mimo_RL_Environment.get_num_actions()
    for i in range(num_steps):
        action_index = randint(0,num_actions)
        ob, reward, gameOver, history = mimo_RL_Environment.step(action_index)
        if gameOver:
            print('Game over! End of episode.')
        print(history)
