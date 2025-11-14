# Representation of Directed Graph as Adjacency Matrix:
# V is number of edges
# edges is in coordinate of (1,0), (2,0), (1,2) 
# if and only if V=3 and 1-0 2-0 1-2

def CreateGraph(V, edges):
    mat = [[0 for _ in range(V)] for _ in range(V)]

    for coordinate in edges:
        u = coordinate[0]
        v = coordinate[1]
        mat[u][v] = 1
        # since this is directed, there is no mat[v][u]
    return mat

V = 3
edges = [[1,0], [2,0], [1,2]]

mat = CreateGraph(V=V, edges=edges)

print("Adjacency matrix for directed graph: ")
for i in range(V):
    for j in range(V):
        print(mat[0][1], end=' ')
    print()
'''
Adjacency matrix for directed graph: 
0 0 0 
0 0 0 
0 0 0 
'''