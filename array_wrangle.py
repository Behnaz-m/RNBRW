import numpy as np
import sys
m = 30201823 
T = 10
address = '/sfs/lustre/bahamut/scratch/bm7mp/git_data_Dec2020/git_'
arg = int(sys.argv[1])# arge is counter for output
#arg = 1 #for the second time when we only have one out put
# T is the number of runs per array, say I have 1000 git out put:git_0-git_999, if T=100 this array adds every 100's runs(10ta 100 taei)and gives 10 outputs

X = np.zeros(m , dtype = int )

# X = np.loadtxt(address+str((arg-1)*T+1) , dtype = int ) 

for i in range((arg-1)*T+1, (arg)*T+1):
    try:
        X += np.loadtxt(address+str(i) , dtype = int )    
    except:
        continue  
        
np.savetxt('/sfs/qumulo/qhome/bm7mp/OS/rnbrw/output/git_total'+str(arg), X.astype(int), fmt='%i') 

