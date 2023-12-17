import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
plt.plot(x, 1 / (1 + np.exp(-x)), label='Sigmoid')
plt.plot(x, np.tanh(x), label='tanh')
plt.plot(x, np.maximum(0, x), label='ReLU')
plt.plot(x, x, label='Identity')
plt.plot(x, np.exp(x) / np.sum(np.exp(x)), label='Softmax')

plt.xlabel('Input')
plt.ylabel('Activation')
plt.title('Activation Functions')
plt.legend()
plt.grid(True)
plt.show()
