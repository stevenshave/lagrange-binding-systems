import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
import numpy
from mpl_toolkits.mplot3d import proj3d


def revenue(h, s):
    return 200*np.power(h, 2/3)*np.power(s, 1/3)


def constraint(h, s):
    return 20000-(h*20+s*170)


fig1 = plt.figure()
ax3d = fig1.add_subplot(111, projection='3d')
x = np.linspace(0, 1000, 500)
y = np.linspace(0, 100, 500)
X, Y = np.meshgrid(x, y)
Z_revenue = np.array(revenue(X, Y)).reshape(X.shape)
Z_constraint = np.array(constraint(np.ravel(X), np.ravel(Y))).reshape(X.shape)
ax3d.plot_surface(X, Y, Z_revenue, cmap="rainbow", antialiased=True)
ax3d.set_xlabel('hours')
ax3d.set_ylabel('steel')
ax3d.set_zlabel('revenue')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
# revenue_contour=ax2.contour(x,y,Z_revenue,[51776.82559821029])
#ax2.clabel(revenue_contour, inline=1, fontsize=10, inline_spacing=0, fmt= '%1.0f')
constraint_contour = ax2.contour(x, y, Z_constraint, [0])
#ax2.clabel(constraint_contour, inline=1, fontsize=10, inline_spacing=0, fmt= '%1.0f')
fig2.suptitle("£20,000 constraint")
ax2.set_xlabel("hours (£20/h)")
ax2.set_ylabel("steel (£170/ton)")

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.set_xlabel("hours (£20/h)")
ax3.set_ylabel("steel (£170/ton)")
fig3.suptitle("Steel factory revenue contours")
revenue_contour = ax3.contour(x, y, Z_revenue)  # [51776.82559821029]
ax3.clabel(revenue_contour, inline=1, fontsize=10, inline_spacing=10, fmt='%i')


fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.set_xlabel("hours (£20/h)")
ax4.set_ylabel("steel (£170/ton)")
fig4.suptitle("Steel factory revenue (£51,777) and constraint contours")
revenue_contour = ax4.contour(x, y, Z_revenue, [51776.82559821029])
constraint_contour = ax4.contour(x, y, Z_constraint, [0], cmap='rainbow')
#ax4.clabel(revenue_contour, inline=1, fontsize=10, inline_spacing=10, fmt='%i')


plt.show()
