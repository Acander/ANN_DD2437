import matplotlib.pyplot as plt
import numpy as np
import pickle


def smothenOutData(data):
    print(len(data))
    window = 20
    for i in range(len(data)):
        s = max(0, i - window)
        e = i+window
        data[i] = np.median(data[s:e])


with open('Lr_Results_Batch_Square.pkl', 'rb') as fp:
    data = pickle.load(fp)

with open('NetworkLoss.pkl', 'rb') as fp:
    data2 = pickle.load(fp)

print(data.keys())
print(data2.keys())

smothenOutData(data[0.001])
smothenOutData(data2[0.001])

data = {'RBF': data[0.001][:12500], 'ANN': data2[0.001][:12500]}

colors = ['red', 'blue', 'green', 'purple', 'black', 'cyan']
for i, k in enumerate(data.keys()):
    d = data[k][0:]
    plt.plot(range(len(d)), d, color=colors[i], linewidth=3, label=str(k))

plt.title = "Residual Error on test set"
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.show()
