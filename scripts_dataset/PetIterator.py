#!/usr/bin/env python
# Este archivo usa el encoding: utf-8
from random import shuffle

class PetIterator:
    def __init__(self,path):
        self.cat_breed = ["Abyssinian",
                          "Bengal",
                          "Birman",
                          "Bombay",
                          "British Shorthair",
                          "Egyptian Mau",
                          "Maine Coon",
                          "Persian",
                          "Ragdoll",
                          "Russian Blue",
                          "Siamese",
                          "Sphynx"]
        self.dog_breed = ["American Bulldog",
                          "American Pit Bull Terrier",
                          "Basset Hound",
                          "Beagle",
                          "Boxer",
                          "Chihuahua",
                          "English Cocker Spaniel",
                          "English Setter",
                          "German Shorthaired",
                          "Great Pyrenees",
                          "Havanese",
                          "Japanese Chin",
                          "Keeshond",
                          "Leonberger",
                          "Miniature Pinscher",
                          "Newfoundland",
                          "Pomeranian",
                          "Pug",
                          "Saint Bernard",
                          "Samoyed",
                          "Scottish Terrier",
                          "Shiba Inu",
                          "Staffordshire Bull Terrier",
                          "Wheaten Terrier",
                          "Yorkshire Terrier"]
        self.cats = {}
        self.dogs = {}
        for i in self.cat_breed:
            self.cats[i] = []
        for i in self.dog_breed:
            self.dogs[i] = []
        
        listTXT = open(path+"/annotations/list.txt","r")
        #primeras 6 lineas no importan
        for i in range(6):
            listTXT.readline()
        for line in listTXT:
            line = line.split(" ")
            #Image CLASS-ID SPECIES BREED ID
            name = line[0]
            isCat = True if int(line[2]) == 1 else False #True cat;False dog
            breed = int(line[3]) - 1
            
            if isCat:
                self.cats[self.cat_breed[breed]].append(name)
            else:
                self.dogs[self.dog_breed[breed]].append(name)

    
    def getCats(self):
        out = []
        for key in self.cats:
            out.extend(self.cats[key])
        shuffle(out)
        return out
    
    def getDogs(self):
        out = []
        for key in self.dogs:
            out.extend(self.dogs[key])
        shuffle(out)
        return out

    def getAll(self):
        out = self.getCats()
        out.extend(self.getDogs())
        shuffle(out)
        return out

#Ejemplo de uso                
#pi = PetIterator("/home/segonzal/Imágenes/Cats&Dogs Dataset")
#for i in pi.getAll():
#    print i
