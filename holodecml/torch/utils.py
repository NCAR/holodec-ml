import torch
import numpy


class Point:
    def __init__(self, coordinates):
        x, y, z, d = coordinates
        self.x = x
        self.y = y
        self.z = z
        self.d = d

def frac_overlap(x, y):
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    return (len(set(x) & set(y))) / len(set(x) | set(y))

def orderless_acc(x, y): # this assumes that x and y are the same sequence length
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    return (len(set(x) & set(y))) / len(x)
        
def distance(p1, p2):
     return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5