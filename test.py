import gym
import highway_env
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from agent import DQNAgent

env = gym.make("highway-v0")
# env.unwrapped.configure(env_config)
env.configure({"duration": 100, "vehicles_count": 40})
# Reset the environment to ensure configuration is applied
env.reset()
dqn_test = DQNAgent(env)
dqn_test.load("models\\DQN_5action_fast_v7.pt")
dqn_test.epsilon = 0

test_len = 10
success_list = np.zeros(test_len)
laneChange_list = np.zeros(test_len)
reward_list = np.zeros(test_len)
steps_list = np.zeros(test_len)

for i in range(test_len):
    done = False
    state = env.reset()
    env.render()
    ep_laneChange = 0  # 每局换道次数
    ep_reward = 0  # 每局回报
    t = 0  # 每局步数
    while not done:
        # action = np.random.randint(3)
        action = dqn_test.choose_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        # obs = np.squeeze(obs)
        # img = Image.fromarray(obs.astype('uint8')).convert('L')
        # img.save('obs.png')
        # print(obs.shape)
        # print(env.action_space.n)
        # plt.pause(0.01)
        # plt.imshow(obs)
        env.render()
        if action == 0 or action == 2:
            ep_laneChange += 1
        ep_reward += reward
        t += 1

    if not info["crashed"]:
        success_list[i] = 1  # 记录本局是否成功无碰撞
    laneChange_list[i] = ep_laneChange  # 记录本局换道数
    reward_list[i] = ep_reward  # 累计回报
    steps_list[i] = t  # 该局步数

print("换道安全率：", sum(success_list)/test_len)
print("平均换道次数：", np.mean(laneChange_list))
print("平均回报：", np.mean(reward_list))
print("平均步数：", np.mean(steps_list))

# python experiments.py evaluate configs/HighwayEnv/env_attention.json \
#                                configs/HighwayEnv/agents/DQNAgent/ego_attention.json \
#                                --train --episodes=4000 --name-from-config
