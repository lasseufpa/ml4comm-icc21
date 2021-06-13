from numpy.random import randint
from beams_calculation import AnalogBeamformer
from render_mimo_rl_simple import Mimo_RL_render
from env_mimo_rl_simple import Mimo_RL_simple

if __name__ == '__main__':
    num_antenna_elements = 32
    analogBeamformer = AnalogBeamformer(num_antenna_elements)
    mimo_RL_Environment = Mimo_RL_simple()
    mimo_RL_render = Mimo_RL_render(analogBeamformer, mimo_RL_Environment)
    mimo_RL_render.should_save_images_as_gif = True #enable saving images in the end

    print('num_antenna_elements=', num_antenna_elements)

    num_steps = 10
    num_actions = mimo_RL_Environment.get_num_actions()
    for i in range(num_steps):
        action_index = randint(0,num_actions)
        ob, reward, gameOver, history = mimo_RL_Environment.step(action_index)
        mimo_RL_render.render()
        if gameOver:
            print('Game over! End of episode.')
        print(history)

    file_name = 'demo_RL.gif'
    mimo_RL_render = mimo_RL_render.save_images_as_gif(file_name)
