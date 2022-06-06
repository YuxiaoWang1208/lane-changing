import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

train_log = open("train_log_v7.txt", 'r')
data = train_log.read().splitlines()
data = list(map(float, data[1:]))
L = len(data)
l = int(L / 5)

ep_count = range(4, l*4+1, 4)  # 横坐标，回合数，数据每4回合记录一次

# data_time_elapsed = data[0::6]
# data_time_ep = [data_time_elapsed[0]]  # 每回合平均实际耗时
# for i in range(1, l):
#     data_time_ep.append((data_time_elapsed[i]-data_time_elapsed[i-1]) / 4)
# plt.figure()
# plt.plot(ep_count, data_time_ep)
# plt.xlabel("Training episodes")
# plt.ylabel("Time elapsed (s)")

data_time_steps = data[0::5]  # 平均每回合运行步数
plt.figure()
plt.plot(ep_count, data_time_steps)
plt.xlabel("Training episodes")
plt.ylabel("Episode steps")

data_rew_mean = data[1::5]  # 每回合对应的平均奖励回报，平均缓冲器大小为10
data_success_mean = data[2::5]  # 每回合对应的平均成功率，运行至仿真回合不撞车即为成功
fig, ax1 = plt.subplots()
ax1.plot(ep_count, data_rew_mean, 'r')
ax1.set_xlabel("Training episodes")
ax1.set_ylabel("Average rewards", color='r')
ax2 = ax1.twinx()
ax2.plot(ep_count, data_success_mean, 'b')
ax2.set_ylabel("Success rate", color='b')

data_speed_mean = data[3::5]  # 每回合对应的智能体平均车速
data_change_count = data[4::5]  # 每回合对应的智能体车道变换次数
fig, ax1 = plt.subplots()
ax1.plot(ep_count, data_speed_mean, 'r')
ax1.set_xlabel("Training episodes")
ax1.set_ylabel("Average speed", color='r')
ax2 = ax1.twinx()
ax2.plot(ep_count, data_change_count, 'b')
ax2.set_ylabel("Lane changing times", color='b')

plt.show()
train_log.close()
