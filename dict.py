try:
    import cPickle as pickle
except:
    import pickle
    
AccInfo={}
startingAccNo=1000

with open('bankaccdb','wb') as file:
    pickle.dump(AccInfo,file)
    pickle.dump(1000,file)
'''
with open('bankaccdb','rb') as file:
    AccInfo=pickle.load(file)
    startingAccNo=pickle.load(file)
    '''
print(AccInfo,startingAccNo)