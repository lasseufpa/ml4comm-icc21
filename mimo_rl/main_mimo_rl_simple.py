from numpy.random import randint
from beamforming_calculation import AnalogBeamformer
from render_mimo_rl_simple import Mimo_RL_render
from env_mimo_rl_simple import Mimo_RL_Simple_Env
from channel_mimo_rl_simple import Grid_Mimo_Channel

if __name__ == '__main__':
    num_antenna_elements=32
    grid_size=6
    analogBeamformer = AnalogBeamformer(num_antenna_elements=num_antenna_elements)
    grid_Mimo_Channel = Grid_Mimo_Channel(num_antenna_elements=num_antenna_elements, grid_size=6)
    mimo_RL_Environment = Mimo_RL_Simple_Env(analogBeamformer, grid_Mimo_Channel)
    mimo_RL_render = Mimo_RL_render(analogBeamformer, mimo_RL_Environment)
    #enable saving images in the end. It is not working now
    mimo_RL_render.should_save_images_as_gif = False 

    print('num_antenna_elements=', num_antenna_elements)

    num_steps = 500
    num_actions = mimo_RL_Environment.get_num_actions()
    for i in range(num_steps):
        action_index = randint(0,num_actions)
        ob, reward, gameOver, history = mimo_RL_Environment.step(action_index)
        mimo_RL_render.render()
        if gameOver:
            print('Game over! End of episode.')
        print(history)

    if mimo_RL_render.should_save_images_as_gif:
        file_name = 'demo_RL.gif'
        mimo_RL_render = mimo_RL_render.save_images_as_gif(file_name)
