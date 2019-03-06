# DWB Mar-5-2019


import numpy as np
import matplotlib.pyplot as plt

import os

plt.style.use('seaborn-whitegrid')

os.chdir('C:/Users/Bynum/Documents/Udacity/Term1/DWB-T1-P3/examples')

x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])

y = np.array([.917,.898,.914,.901,.917,.901,.897,.906,.896,.908,.897,.908,.899,.924])



plt.plot(x,y,'o',color='black')
plt.title('Project Training Accuracy Trail Runs')
plt.ylim(0.8,1)
plt.xlabel('Trail/Iteration Number')
plt.ylabel('Accuracy Measurement - Test Data Set')
plt.savefig('accuracy.png')

plt.show()
