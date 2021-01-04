import numpy as np
import sys
m = 150580575 
T = 9
address = '/sfs/qumulo/qhome/bm7mp/OS/rnbrw/output/git_full_total'
#arg = int(sys.argv[1])# arge is counter for output
arg = 2 #for the second time when we only have one out put
# T is the number of runs per array, say I have 1000 git out put:git_0-git_999, if T=100 this array adds every 100's runs(10ta 100 taei)and gives 10 outputs

X = np.zeros(m , dtype = int )

# X = np.loadtxt(address+str((arg-1)*T+1) , dtype = int ) 

for i in range((arg-1)*T+1, (arg)*T+1):
    try:
        X += np.loadtxt(address+str(i) , dtype = int )    
    except:
        continue  
        
np.savetxt('/sfs/qumulo/qhome/bm7mp/OS/rnbrw/output/git_ful_total', X.astype(int), fmt='%i') 

