from hash import Node, Graph, hash_graph
from typing import Tuple
import numpy as np
import pandas as pd

Max_Length = 20

class Tree:
    def __init__(self, root: Node, graph: Graph):
        self.root = root
        self.graph = graph
        self.size = len(graph.nodes)

def merge(left: Tree, right: Tree, op: str) -> Tree:
    root = Node(left.size + right.size, op)
    for i in range(len(right.graph.nodes)):
        right.graph.nodes[i].id += left.size
    right.root.id += left.size
    root.edges = [left.root, right.root]
    left.root.edges.append(root)
    right.root.edges.append(root)
    merged_nodes = left.graph.nodes + right.graph.nodes + [root]
    merged_graph = Graph(merged_nodes)
    return Tree(root, merged_graph)

def generate_seq(maxlen: int) -> Tuple[str, Tree, int]:
    if maxlen == 1:
        op = np.random.choice(['1', '0'])
        return op, Tree(Node(0, op), Graph([Node(0, op)])), int(op)
    if np.random.randint(low = 0, high = maxlen * maxlen):
        op = np.random.choice(['~', '|', '&'])
        if maxlen == 2:
            op = '~'
        if maxlen == 3:
            temp1 = np.random.choice(['1', '0'])
            temp2 = np.random.choice(['1', '0'])
            op = np.random.choice(['|', '&'])
            return temp1 + temp2 + op, merge(Tree(Node(0, temp1), Graph([Node(0, temp1)])), Tree(Node(0, temp2), Graph([Node(0, temp2)])), op), int(temp1) | int(temp2) if op == '|' else int(temp1) & int(temp2)
        if op == '~':
            gen = generate_seq(maxlen - 1)
            tree = gen[1]
            root = Node(tree.size, '~')
            root.edges = [tree.root]
            tree.root.edges.append(root)
            new_graph = Graph(tree.graph.nodes + [root])
            new_tree = Tree(root, new_graph)
            return gen[0] + '~', new_tree, 1 - gen[2]
        else:
            left_len = np.random.randint(1, maxlen - 2)
            right_len = maxlen - 1 - left_len
            left = generate_seq(left_len)
            right = generate_seq(right_len)
            if op == '|':
                ans = left[2] | right[2]
            else:
                ans = left[2] & right[2]
            return left[0] + right[0] + op, merge(left[1], right[1], op), ans
    else:
        op = np.random.choice(['1', '0'])
        return op, Tree(Node(0, op), Graph([Node(0, op)])), int(op)

def translate(tree: Tree) -> str:
    if tree.size == 1:
        return tree.root.value
    def dfs(node: Node, parent: Node) -> str:
        if len(node.edges) == 1 and node.edges[0] == parent:
            return node.value
        res = ''
        for neighbor in node.edges:
            if neighbor != parent:
                res += node.value
                res += dfs(neighbor, node)
        if node.value != '~':
            res = res[1:]
            if parent is not None:
                res = '(' + res + ')'
        elif len(res) > 2:
            res = res[1:]
            res = '~(' + res + ')'
        return res
    res = dfs(tree.root, None)
    def off_brackets(s: str) -> str:
        and_place = -1
        or_place = -1
        cnt = 0
        for i in range(len(s)):
            if s[i] == '(':
                cnt += 1
            elif s[i] == ')':
                cnt -= 1
            elif cnt == 0:
                if s[i] == '&':
                    and_place = i
                elif s[i] == '|':
                    or_place = i
        if or_place == -1 and and_place == -1:
            if len(s) < 3:
                return s
            for i in range(len(s)):
                if s[i] != '~':
                    return s[:i] + off_brackets(s[i + 1:-1])
        if and_place == -1:
            return off_brackets(s[:or_place]) + '|' + off_brackets(s[or_place + 1:])
        if or_place == -1:
            left = s[:and_place]
            right = s[and_place + 1:]
            cnt = 0
            have_and = 0
            for i in range(len(left)):
                if left[i] == '(':
                    cnt += 1
                elif left[i] == ')':
                    cnt -= 1
                elif cnt == 1 and left[i] == '&':
                    have_and = 1
                    break
            if have_and:
                left = off_brackets(left)
            else:
                if len(left) > 2:
                    for i in range(len(left)):
                        if left[i] != '~':
                            left = left[:i] + '(' + off_brackets(left[i + 1:-1]) + ')'
                            break
            cnt = 0
            have_and = 0
            for i in range(len(right)):
                if right[i] == '(':
                    cnt += 1
                elif right[i] == ')':
                    cnt -= 1
                elif cnt == 1 and right[i] == '&':
                    have_and = 1
                    break
            if have_and:
                right = off_brackets(right)
            else:
                if len(right) > 2:
                    for i in range(len(right)):
                        if right[i] != '~':
                            right = right[:i] + '(' + off_brackets(right[i + 1:-1]) + ')'
                            break
            return left + '&' + right
    return off_brackets(res)

Train_data = []
Test_data = []
for _ in range(10000):
    seq, tree, ans = generate_seq(Max_Length)
    flag1 = 1
    flag2 = 1
    for i in range(len(tree.graph.nodes)):
        hash = hash_graph(tree.graph, i)
        hash = int(hash, 16)
        if hash % 79 == 42:
            flag1 = 0
    if flag1:
        Train_data.append([translate(tree), ans])
    else:
        Test_data.append([translate(tree), ans])

np.random.shuffle(Train_data)
np.random.shuffle(Test_data)
pd.DataFrame(Train_data, columns = ['seq', 'ans']).to_csv('2025-12月/数据生成-子图哈希法/train.csv', index = False)
pd.DataFrame(Test_data, columns = ['seq', 'ans']).to_csv('2025-12月/数据生成-子图哈希法/test.csv', index = False)
if __name__ == "__main__":
    print("Train and Test data generated and saved to CSV files.")
    print("Train data size:", len(Train_data))
    print("Test data size:", len(Test_data))