import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
for y in range(0,240,30):
    print y
    ax1.add_patch(patches.Rectangle((0,y),45,9, facecolor='white'))
    plt.text(0, y, 'lalala')
    ax1.add_patch(patches.Rectangle((0, y+11), 45, 9, facecolor='white'))

    plt.text(0, y, 'lalalalalalala', fontsize=4)
for y in range(0,240,60):
    print y
    ax1.add_patch(patches.Rectangle((50,y+15.5),45,9))
    ax1.add_patch(patches.Rectangle((50, y+11+15.5), 45, 9))

ax1.axis([0.0,150, 0.0,250])
ax1.set_autoscale_on(False)

plt.show(ax1)