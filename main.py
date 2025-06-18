import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint


def odesys(y, t, m, c, k1, k2, l, g):  # функция системы уравнений
    # y[0,1,2,3] = phi,tetta,phi',tetta'
    # dy[0,1,2,3] = phi',tetta',phi'',tetta''

    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = 1
    a12 = -1 * np.cos(y[0] - y[1])
    a21 = -1 * np.cos(y[0] - y[1])
    a22 = 1

    b1 = (g / l) * np.sin(y[0]) + np.sin(y[0] - y[1]) * (y[3] ** 2) - ((c * y[0] + k1 * y[2]) / (m * l ** 2))
    b2 = -1 * (g / l) * np.sin(y[1]) - (k2 * y[3]) / (m * l ** 2) - np.sin(y[0] - y[1]) * (y[2] ** 2)

    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21)
    dy[3] = (b2 * a11 - b1 * a21) / (a11 * a22 - a12 * a21)

    return dy


# задаём все параметры
m = 0.24
c = 2
k1 = 0.1
k2 = 0.1
l = 1
g = 9.81
t_fin = 20

t = np.linspace(0, t_fin, 1001)

# задаём начальное состояние
phi0 = 1.0472
tetta0 = -1.4
dtetta0 = 0
dphi0 = 0
t0 = 0

y0 = [phi0, tetta0, dphi0, dtetta0]  # вектор начального состояния

Y = odeint(odesys, y0, t, (m, c, k1, k2, l, g))

phi = Y[:, 0]  # получили решение
tetta = Y[:, 1]
dphi = Y[:, 2]
dtetta = Y[:, 3]

ddphi = np.array([odesys(yi, ti, m, c, k1, k2, l, g)[2] for yi, ti in zip(Y, t)])
ddtetta = np.array([odesys(yi, ti, m, c, k1, k2, l, g)[3] for yi, ti in zip(Y, t)])

Rx = (m / l) * (ddtetta * np.cos(tetta) - np.sin(tetta) * dtetta ** 2 - ddphi * np.cos(phi) + np.sin(phi) * dphi ** 2)
Ry = (m / l) * (ddtetta * np.sin(tetta) + np.cos(tetta) * dtetta ** 2 - ddphi * np.sin(phi) - np.cos(
    phi) * dphi ** 2) + m * g

fig_for_graphs = plt.figure(figsize=[13, 7])  # построим их графики
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(t, phi, color='blue')
ax_for_graphs.set_title("phi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(t, tetta, color='red')
ax_for_graphs.set_title('tetta(t)')
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs.plot(t, dphi, color='green')
ax_for_graphs.set_title("phi'(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 4)
ax_for_graphs.plot(t, dtetta, color='black')
ax_for_graphs.set_title("tetta'(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

fig_for_grap = plt.figure(figsize=[10, 5])  # построим их графики
ax_for_grap = fig_for_grap.add_subplot(2, 2, 1)
ax_for_grap.plot(t, Rx, color='blue')
ax_for_grap.set_title("Rx(t)")
ax_for_grap.set(xlim=[0, t_fin])
ax_for_grap.grid(True)

ax_for_grap = fig_for_grap.add_subplot(2, 2, 2)
ax_for_grap.plot(t, Ry, color='blue')
ax_for_grap.set_title("Ry(t)")
ax_for_grap.set(xlim=[0, t_fin])
ax_for_grap.grid(True)

x0 = 1  # положение равновесия пружинки
xA = x0 + 0.2
yA = x0
xB = xA - l * np.sin(phi)
yB = yA + l * np.cos(phi)
xC = xB + l * np.sin(tetta)
yC = yB - l * np.cos(tetta)
xP = xA - 0.05
yP = yA - 0.05
xN = xA + 0.05
yN = yA - 0.05

N = 2
r1 = 0.005
r2 = 0.1

thetta = np.linspace(0, N * 6.28 - phi[0], 100)
xS = -(r1 + thetta * (r2 - r1) / thetta[-1]) * np.sin(thetta)
yS = (r1 + thetta * (r2 - r1) / thetta[-1]) * np.cos(thetta)

fig = plt.figure(figsize=[13, 9])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[-0.25, 3], ylim=[-0.25, 2])


BC = ax.plot([xB[0], xC[0]], [yB[0], yC[0]], color=[1, 0, 0])[0]  # 2палка
AB = ax.plot([xA, xB[0]], [yA, yB[0]], color=[1, 0, 0])[0]  # 1палка
AP = ax.plot([xA, xP], [yA, yP], color=[0, 0, 0])[0]  # опора
AN = ax.plot([xA, xN], [yA, yN], color=[0, 0, 0])[0]  # опора
A = ax.plot(xA, yA, 'o', color=[1, 0, 0])[0]
B = ax.plot(xB[0], yB[0], 'o', color=[0, 1, 0])[0]
C = ax.plot(xC[0], yC[0], 'o', color=[0, 1, 0])[0]
PN = ax.plot([xP - 0.01, xN + 0.01], [yP, yN], color=[0, 0, 0])[0]  # опора
Spiral = ax.plot(xS + xA, yS + yA, color=[1, 0, 0])[0]

def kadr(i):
    B.set_data(xB[i], yB[i])
    C.set_data(xC[i], yC[i])
    AB.set_data([xA, xB[i]], [yA, yB[i]])
    BC.set_data([xB[i], xC[i]], [yB[i], yC[i]])
    thetta = np.linspace(0, N * 6.28 - phi[i], 100)
    xS = -(r1 + thetta * (r2 - r1) / thetta[-1]) * np.sin(thetta)
    yS = (r1 + thetta * (r2 - r1) / thetta[-1]) * np.cos(thetta)
    Spiral.set_data(xS + xA, yS + yA)
    return [A, B, AB, BC, Spiral]


kino = FuncAnimation(fig, kadr, interval=10, frames=len(t))

plt.show()
