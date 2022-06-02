import os
import sys
import numpy as np
import gym
import highway_env
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import *


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["SDL_VIDEODRIVER"] ="dummy"  #华为云训练和测试需要
# hyper-parameters
# 设置超参数
TARGET_FREQ = 40  # target network update frequency
GAMMA = 0.95
INITIAL_EPSILON = 0.7
MIN_EPSILON = 0.01
DECAY = 0.996
LR = 5e-4
MEMORY_SIZE = 1000
EXPLORE = 10000
LEARN_FREQ = 4
BATCH_SIZE = 2

class ReplayBuffer(object):
    # 在此编写经验回放
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience):
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > self.size():
            batch_size = self.size()
        if continuous:
            rand = np.random.randint(0, self.size()-batch_size)
            return [self.buffer[i] for i in range(rand, rand+batch_size)]
        else:
            indexes = np.random.choice(np.arange(self.size()), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

# class EgoAttention(nn.Module):
#    #如果要用attention，在此编写Attention

class DQNAgent:
    def __init__(self, env, memory_size=MEMORY_SIZE, epsilon=INITIAL_EPSILON, lr=LR):
        self.net = DQN()
        self.target_net = DQN()
        self.env = env
        self.epsilon = epsilon
        self.memory_buffer = ReplayBuffer(memory_size)
        self.action_space = self.env.action_space
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.learn_step_counter = 0

    # def choose_action(self, state):
    #     x = state.flatten()
    #     x = torch.FloatTensor(x)
    #     if np.random.uniform() < 1 - self.epsilon:
    #         actions_value = self.net.forward(x)  # 计算状态行动价值
    #         action = torch.max(actions_value, -1)[1].data.numpy()
    #         action = action.max()
    #     else:
    #         action = np.random.randint(0, self.action_space.n)
    #     return action

    def choose_action(self, state):
        x = state.flatten()
        x = torch.FloatTensor(x)
        if np.random.uniform() < 1 - self.epsilon:
            actions_value = self.net.forward(x).tolist()  # 计算状态行动价值
            available_actions = self.env.rule()
            available_actions_value = []
            l_a = len(available_actions)
            for i in range(l_a):
                available_actions_value.append(actions_value[available_actions[i]])
            action = actions_value.index(max(available_actions_value))
            # action = torch.max(actions_value, -1)[1].data.numpy()
            # action = action.max()
        else:
            available_actions = self.env.rule()
            action_index = np.random.randint(0, len(available_actions))
            action = available_actions[action_index]
        return action

    def learn(self, batch_size):
        if self.memory_buffer.size() > batch_size:
            if self.learn_step_counter % TARGET_FREQ == 0:
                self.target_net.load_state_dict(self.net.state_dict())
            self.learn_step_counter += 1

            batch = self.memory_buffer.sample(batch_size, False)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.FloatTensor(states)# .to(device)
            actions = torch.FloatTensor(actions)# .to(device)
            rewards = torch.FloatTensor(rewards)# .to(device)
            next_states = torch.FloatTensor(next_states)# .to(device)
            dones = torch.FloatTensor(dones)

            with torch.no_grad():
                q_next = self.target_net.forward(next_states)
                q_target = rewards + (1-dones) * GAMMA * torch.max(q_next, dim=1, keepdim=True)[0]
            q = self.net.forward(states)
            q = q.squeeze(1).gather(1, actions.unsqueeze(1).to(torch.int64))
            loss = self.loss_func(q, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss
        else:
            return 0

    def save(self, PATH):
        torch.save(self.net.state_dict(), PATH)

    def load(self, PATH):
        self.net.load_state_dict(torch.load(PATH))
        self.net.eval()

'''
class DDQN(object):
    def __init__(self, num_states, num_actions, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 编写初始化

    def choose_action(self, state):
        #编写选择动作
        return action



    def learn(self):
       #编写agent算法训练过程

    def save(self, directory, i):
        torch.save(self.eval_net.state_dict(), directory + 'dqn{}.pth'.format(i))
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self, directory, i):
        self.eval_net.load_state_dict(torch.load(directory + 'dqn{}.pth'.format(i)))
        print("====================================")
        print("Model has been loaded...")
        print("====================================")
'''


def train(args=None):
    # if args is None:
    #     args = sys.argv[1:]
    # args = parse_args(args)

    # Check if a GPU ID was set
    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # world, client = PlayGame.setup_world(host='localhost', fixed_delta_seconds=0.05, reload=True)
    # # client.set_timeout(5.0)
    # if world is None:
    #     return
    # traffic_manager = client.get_trafficmanager(8000)

    env_config = {
                "id": "highway-v0",
                "import_module": "highway_env",
                "lanes_count": 3,
                "vehicles_count": 40,   # 环境车数量
                "duration": 50,         #每个episode的step数
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 5,  # 15
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
    env = gym.make("highway-fast-v0")
    env.unwrapped.configure(env_config)
    # env = gym.make("CartPole-v0")
    # env = env.unwrapped
    env.reset()
    # 编写算法训练过程

    # directory = './weights_with_ego_attention/'
    # dqn.writer = SummaryWriter(directory)
    print(env.action_space.n)
    dqn = DQNAgent(env)
    episodes = 1000  #尝试不同episodes结果   1000-2000
    print("Collecting Experience....")


    # 打开记录指标文件
    log_file = open("train_log_v1.txt", 'w')
    log_file.write("avg_steps, avg_reward, "
                   "success_rate, avg_speed, avg_laneChange\n")
    # 初始化指标记录容器
    avg_len = 10  # 求平均值的窗长为10
    avg_counter = 0
    steps_list = np.zeros(avg_len)
    reward_list = np.zeros(avg_len)
    success_eps = 0
    speed_list = np.zeros(avg_len)
    laneChange_list = np.zeros(avg_len)
    for i in range(episodes):
        print("Episode ", i + 1)
        state = env.reset()
        env.render()
        done = False
        dqn.epsilon = max(INITIAL_EPSILON * DECAY ** i, MIN_EPSILON)
        # for t in count():
        # for t in range(300):
        # 每局内需要记录的指标
        t = 0
        ep_reward = 0
        ep_speed = 0  # 每局平均速度
        ep_laneChange = 0  # 每局换道次数
        while not done:
            action = dqn.choose_action(state)
            # action = 1
            next_state, reward, done, info = env.step(action)
            env.render()
            dqn.memory_buffer.add((state.flatten(), action, reward, next_state.flatten(), done))
            state = next_state
            # 需要记录的指标
            ep_reward += reward
            ep_speed += info["speed"]
            if action == 0 or action == 2:
                ep_laneChange += 1
            if t % LEARN_FREQ == 0:
                loss = dqn.learn(BATCH_SIZE)
            t += 1

        steps_list[avg_counter] = t  # 该局步数
        reward_list[avg_counter] = ep_reward  # 累计回报
        print(ep_reward)
        if not info["crashed"]:
            success_eps += 1  # 平均成功率
        ep_speed = ep_speed / t  # 平均速度
        speed_list[avg_counter] = ep_speed
        laneChange_list[avg_counter] = ep_laneChange
        avg_counter += 1
        if (i + 1) % avg_len == 0:
            # 记录平均指标数据
            avg_counter = 0
            avg_steps = np.mean(steps_list)
            log_file.write(str(avg_steps) + '\n')
            avg_reward = np.mean(reward_list)
            log_file.write(str(avg_reward) + '\n')
            log_file.write(str(success_eps / avg_len) + '\n')
            success_eps = 0
            avg_speed = np.mean(speed_list)
            log_file.write(str(avg_speed) + '\n')
            avg_laneChange = np.mean(laneChange_list)
            log_file.write(str(avg_laneChange) + '\n')

    # if (i + 1) % 100 == 0:
    #     dqn.save("models\\DQN_3action_{}.pt".format((i+1)//100))
    dqn.save(".\\models\\DQN_3action_fast_v1.pt")
    log_file.close()

if __name__ == "__main__":
    train()
    # test()
