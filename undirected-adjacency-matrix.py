# Representation of Undirected Graph as Adjacency Matrix:
# V is number of edges
# edges is in coordinate of (0,1), (0,2), (1,2) 
# if and only if V=3 and 0-1 0-2 1-2
def CreateGraph(V, edges):
    mat = [[0 for _ in range(V)] for _ in range(V)]

    for edge in edges:
        u = edge[0]
        v = edge[1]
        mat[u][v] = 1

        # since the graph is undirected:
        mat[v][u] = 1
    return mat

V = 3

# list of edges (u, v)
edges = [[0, 1], [0, 2], [1, 2]]

# build the graph using edges
mat = CreateGraph(V, edges=edges)

print("Adjacency Matrix Representation: ")
for i in range(V):
    for j in range(V):
        print(mat[i][j], end=' ')
    print()

'''
Adjacency Matrix Representation: 
0 1 1 
1 0 1 
1 1 0 
'''