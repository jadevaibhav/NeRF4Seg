import numpy as np

a = np.load('masks_8.npy').reshape(-1)
print(a.shape)
a1 = np.unique(a)
print(a1)
print(a1.shape)
#b = np.zeros(a.shape)
for i in range(a.shape[0]):
    #print(np.where(a1==a[i])[0])
    b[i] = np.where(a1==a[i])[0]

print(np.unique(b))
np.save('masks_8.npy',b)