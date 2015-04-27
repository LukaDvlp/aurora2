#!/usr/bin/env python
""" A* shortest path algorithm

Usage:
    $ python astar.py

Original implementation by Justin Poliey <justin.d.poliey@gmail.com>
https://gist.github.com/jdp/1687840#file_astar.py

Modified by Kyohei Otsu <kyon@ac.jaxa.jp> since 2015-04-21
"""

from math import sqrt
from itertools import product

DANGER = 100

class AStar(object):
    def __init__(self, graph):
        self.graph = graph
        
    def heuristic(self, node, start, end):
        raise NotImplementedError
        
    def search(self, start, end):
        if end.cost >= DANGER: return None, None
        openset = set()
        closedset = set()
        current = start
        openset.add(current)
        while openset:
            current = min(openset, key=lambda o:o.g + o.h)
            if current == end:
                path = []
                while current.parent:
                    path.append(current)
                    current = current.parent
                path.append(current)
                return path[::-1], end.parent.g
            openset.remove(current)
            closedset.add(current)
            for node in self.graph[current]:
                if node in closedset:
                    continue
                if node in openset:
                    new_g = current.g + current.move_cost(node) + current.node_cost(node)
                    if node.g > new_g:
                        node.g = new_g
                        node.parent = current
                else:
                    if current.node_cost(node) < DANGER:
                        node.g = current.g + current.move_cost(node) + current.node_cost(node)
                        node.h = self.heuristic(node, start, end)
                        node.parent = current
                        openset.add(node)
        return None, None
 
 
class AStarNode(object):
    def __init__(self):
        self.g = 0
        self.h = 0
        self.parent = None
        
    def move_cost(self, other):
        raise NotImplementedError

    def node_cost(self, other):
        raise NotImplementedError


class AStarGrid(AStar):
    def heuristic(self, node, start, end):
        return sqrt((end.x - node.x)**2 + (end.y - node.y)**2)
 
 
class AStarGridNode(AStarNode):
    def __init__(self, x, y, cost=0):
        self.x, self.y = x, y
        self.cost = cost
        super(AStarGridNode, self).__init__()
 
    def move_cost(self, other):
        diagonal = abs(self.x - other.x) == 1 and abs(self.y - other.y) == 1
        #return 14 if diagonal else 10
        long_diagonal = abs(self.x - other.x) == 2 or abs(self.y - other.y) == 2
        return (22 if long_diagonal else 14 if diagonal else 10)
    
    def node_cost(self, other):
        return DANGER * other.cost



def make_graph(width, height, costs=None):
    nodes = [[AStarGridNode(x, y) for y in range(height)] for x in range(width)]
    graph = {}
    for x, y in product(range(width), range(height)):
        node = nodes[x][y]
        if costs is not None: node.cost = costs[x][y]
        graph[node] = []
        #for i, j in product([-1, 0, 1], [-1, 0, 1]):
        for i, j in product([-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]):
            if not (0 <= x + i < width):
                continue
            if not (0 <= y + j < height):
                continue
            if abs(i) == 2 and abs(j) == 2: continue
            if abs(i) == 0 and abs(j) == 2: continue
            if abs(i) == 2 and abs(j) == 0: continue
            graph[nodes[x][y]].append(nodes[x+i][y+j])
    return graph, nodes
 

####################################
#  sample code                     #
####################################
if __name__ == '__main__':
    graph, nodes = make_graph(8, 8)
    paths = AStarGrid(graph)
    start, end = nodes[1][1], nodes[6][7]
    path = paths.search(start, end)
    if path is None:
        print "No path found"
    else:
        print "Path found:"
        for p in path:
            print p.x, p.y


