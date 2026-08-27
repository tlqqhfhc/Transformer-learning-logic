""" hash.py
    对图进行哈希
"""
import hashlib
from queue import Queue

MaxDeepth = 3

class Node:
    def __init__(self, id, value):
        self.id = id
        self.value = value
        self.edges = []

class Graph:
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes

def hash_graph(graph: Graph, k):
    queue = Queue()
    queue.put(graph.nodes[k])
    visited = []
    deepth = {}
    deepth[graph.nodes[k]] = 0
    string = ''
    while queue.qsize() > 0:
        now = queue.get()
        string += str(now.value) + str(deepth[now])
        visited.append(now)
        for neighbor in now.edges:
            if neighbor not in visited:
                deepth[neighbor] = deepth[now] + 1
                if deepth[neighbor] <= MaxDeepth:
                    queue.put(neighbor)
    hash_object = hashlib.md5(string.encode())
    return hash_object.hexdigest()

if __name__ == "__main__":
    # 创建一个简单的图进行测试
    node0 = Node(0, '|')
    node1 = Node(1, '~')
    node2 = Node(2, '0')
    node3 = Node(3, '1')

    node0.edges = [node1, node2]
    node1.edges = [node0, node3]
    node2.edges = [node0]
    node3.edges = [node1]

    graph = Graph([node0, node1, node2, node3])

    print("Graph Hash:", hash_graph(graph, 1))