import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# 目標函數
def fitness(x1, x2, x3, target):
    c1 = np.cos(3 * np.pi * x1)
    c2 = np.cos(4 * np.pi * x2)
    n = 16000
    y = x1**2 + x2**2 - 0.3*c1 - 0.4*c2 + 0.7
    return 0.5 + (np.sin(x1**2 - x2**2) - 0.5) / (1 + 0.001 * (x1**2 + x2**2 + x3**2))**2

#更新粒子的位置
def update_pos(x1, x2, x3, vx1, vx2, vx3):
    new_x1 = x1 + vx1
    new_x2 = x2 + vx2
    new_x3 = x3 + vx3
    return new_x1, new_x2, new_x3

# 更新粒子的速度
def update_vel(x1, x2, x3, vx1, vx2, vx3, p, g, w=0.5,random_max=0.14):  # p = 哪個粒子 g = 全體最佳 w = 權重 random_max = 隨機值最大範圍
    ran1 = random.uniform(0, random_max)
    ran2 = random.uniform(0, random_max)
    new_vx1 = w * vx1 + ran1 * (p["x1"] - x1) + ran2 * (g["x1"] - x1)
    new_vx2 = w * vx2 + ran1 * (p["x2"] - x2) + ran2 * (g["x2"] - x2)
    new_vx3 = w * vx3 + ran1 * (p["x3"] - x3) + ran2 * (g["x3"] - x3)
    return new_vx1, new_vx2, new_vx3


#粒子的數量、範圍設定
N = 100
x1_min, x1_max = -100,100
x2_min, x2_max = -100,100
x3_min, x3_max = -100,100

target = 0

#初始化粒子資訊
ps = [{"x1": random.uniform(x1_min, x1_max), "x2": random.uniform(x2_min, x2_max),"x3": random.uniform(x3_min, x3_max)} for i in range(N)]  # 隨機產生粒子的座標
vs = [{"x1": 0.0, "x2": 0.0, "x3": 0.0} for i in range(N)]  # 粒子的速度
pbest_positions = ps
pbest_score = [fitness(p["x1"], p["x2"], p["x3"], target) for p in ps]
best_particle = np.argmin(pbest_score)
gbest_position = pbest_positions[best_particle]

# df1 = pd.DataFrame(ps)
# print('粒子初始位置')
# print(df1.head())
# print(df1.tail())
#
# df2 = pd.DataFrame(vs)
# print('粒子初始速度')
# print(df2.head())
# print(df2.tail())

# 設定圖型
# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(211, projection='3d')
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')
# ax.set_zlabel('$F(x)$')
#
# x1_coord = np.linspace(x1_min, x1_max, 100)
# x2_coord = np.linspace(x2_min, x2_max, 100)
# X, Y = np.meshgrid(x1_coord, x2_coord)

# ax.plot_wireframe(X, Y, fitness(X, Y), color='b', rstride=2, cstride=2, linewidth=0.3)
# ims = []
best_scores = []

T = 500
for t in range(T):#迭代次數
    # title = 'iter = {:02}'.format(t + 1)
    # ax.title.set_text(title)
    for i in range(N):#更新N個粒子
        x1, x2, x3 = ps[i]["x1"], ps[i]["x2"], ps[i]["x3"]  # 粒子的資訊
        vx1, vx2, vx3 = vs[i]["x1"], vs[i]["x2"], vs[i]["x3"]  # 粒子的速度
        p = pbest_positions[i]  # 哪顆粒子

        # 更新粒子
        new_x1, new_x2, new_x3 = update_pos(x1, x2, x3, vx1, vx2, vx3)
        ps[i] = {"x1": new_x1, "x2": new_x2, "x3": new_x3}

        new_vx1, new_vx2, new_vx3 = update_vel(x1, x2, x3, vx1, vx2, vx3, p, gbest_position)
        vs[i] = {"x1": new_vx1, "x2": new_vx2, "x3": new_vx3}

        score = fitness(new_x1, new_x2, new_x3, target)
        if score < pbest_score[i]:
            pbest_score[i] = score
            pbest_positions[i] = {"x1": new_x1, "x2": new_x2, "x3": new_x3}

        best_particle = np.argmin(pbest_score)
        gbest_position = pbest_positions[best_particle]
    best_scores.append(fitness(gbest_position["x1"], gbest_position["x2"], gbest_position["x3"], target))

    # im = ax.scatter3D([ps[i]["x1"] for i in range(N)], [ps[i]["x2"] for i in range(N)]
    #                   , [fitness(ps[i]["x1"], ps[i]["x2"]) for i in range(N)], c='r')
    # ims.append([im])

# ani = animation.ArtistAnimation(fig, ims, repeat=True, interval=150)
#
# bx = fig.add_subplot(212)
# bx.title.set_text('best score curve')
# bx.set_xlabel('iter times')
# bx.set_ylabel('best score')
# plt.plot(best_scores)
# plt.show()
#ani.save('PSO.gif', writer='Pillow')
# 最適解
print(gbest_position)
print(min(pbest_score))
print(best_scores)

# df1 = pd.DataFrame(ps)
# print('粒子最終位置')
# print(df1.head())
# print(df1.tail())
#
# df2 = pd.DataFrame(vs)
# print('粒子最終速度')
# print(df2.head())
# print(df2.tail())