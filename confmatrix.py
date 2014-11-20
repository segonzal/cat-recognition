import numpy as np


def getMAT():
    return {("cat","cat") : 0,
        ("cat","dog") : 0,
        ("dog","cat") : 0,
        ("dog","dog") : 0}

cmat = {}

def processFile(inFile,thetas):
    #header doesn't help us
    inFile.readline()
    # now read what is important
    
    #image counter - over 200 are no-dogs
    j = 0
    for line in inFile:
        line = line.split();
        
        #1 : gato
        #0 : perro
        given_class = int(line[0])
        
        dog_P = float(line[1])
        nodog_P = float(line[2])
        
        # model says is a dog
        for i in range(len(thetas)):
            if dog_P / nodog_P > thetas[i]:
                if j < 200:#is REALLY a dog?
                    if given_class == 0:#es perro
                        cmat[thetas[i]][("dog","dog")] += 1
                    else:
                        cmat[thetas[i]][("dog","cat")] += 1
                else:
                    if given_class == 0:#es perro
                        cmat[thetas[i]][("cat","dog")] += 1
                    else:
                        cmat[thetas[i]][("cat","cat")] += 1
        j += 1

def writeFile(thetas,outfile):
    for key in cmat:
        outfile.write(str(key)+"\t"+str(cmat[key])+"\n")

files = [
        "results1000"
        ]

thetas = np.arange(0.1, 1, 0.05).tolist()

for f in files:
    inF = open(f,'r')
    outF = open(f+"-cmat",'w')
    for i in range(len(thetas)):
        cmat[thetas[i]] = getMAT()
    processFile(inF,thetas)
    writeFile(thetas,outF)
    outF.close()
    inF.close()
