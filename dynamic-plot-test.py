import matplotlib.pyplot as plt
import numpy as np
import time

# Enable interactive mode
plt.ion()

# Initial plot setup
fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
line, = ax.plot(x, y)

# Function to update the plot
def update_plot(new_y):
    line.set_ydata(new_y)
    fig.canvas.draw()
    fig.canvas.flush_events()

# Simulate dynamic data update
for i in range(50):
    new_y = np.sin(x + i/10.0)
    update_plot(new_y)
    time.sleep(0.1)

# Keep the plot open after the updates
plt.ioff()
plt.show()
