import gym
import highway_env
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from agent import DQNAgent

env_config = {
    "id": "highway-v0",
    "import_module": "highway_env",
    "lanes_count": 3,
    "vehicles_count": 40,
    "duration": 100,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    # "centering_position": [0.3, 0.5],
    # "observation": {
    #             # "type": "GrayscaleObservation",
    #             "weights": [0.2989, 0.5870, 0.1140],  #weights for RGB conversion,
    #             "stack_size": 4,
    #             "observation_shape": (150, 600)  # (150, 600)
    #             },
    # "observation": {
    #     "type": "OccupancyGrid",
    #     "vehicles_count": 15,
    #     "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    #     "features_range": {
    #         "x": [-100, 100],
    #         "y": [-100, 100],
    #         "vx": [-20, 20],
    #         "vy": [-20, 20]
    #     },
    #     "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
    #     "grid_step": [5, 5],
    #     "absolute": False
    # },
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        # "absolute": True,
        "order": "shuffled"
    },
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": False
    },
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    # "destination": "o1"
}

env = gym.make("highway-v0")
env.unwrapped.configure(env_config)
# Reset the environment to ensure configuration is applied
env.reset()
dqn_test = DQNAgent(env)
dqn_test.load("models\\DQN_3action_fast_v0.pt")
dqn_test.epsilon = 0

for i in range(10):
    done = False
    state = env.reset()
    env.render()
    while not done:
        # action = np.random.randint(3)
        action = dqn_test.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        # obs = np.squeeze(obs)
        # img = Image.fromarray(obs.astype('uint8')).convert('L')
        # img.save('obs.png')
        # print(obs.shape)
        # print(env.action_space.n)
        # plt.pause(0.01)
        # plt.imshow(obs)
        env.render()


# python experiments.py evaluate configs/HighwayEnv/env_attention.json \
#                                configs/HighwayEnv/agents/DQNAgent/ego_attention.json \
#                                --train --episodes=4000 --name-from-config
