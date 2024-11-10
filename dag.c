#include <stdio.h>
#include <limits.h>

#define MAX 100
#define INF INT_MAX

void printMatrix(int matrix[MAX][MAX], int V) {
    printf("Adjacency Matrix:\n");
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (matrix[i][j] == INF) {
                printf("INF ");
            } else {
                printf("%d ", matrix[i][j]);
            }
        }
        printf("\n");
    }
}

void topologicalSort(int v, int visited[], int stack[], int *top, int graph[MAX][MAX], int V) {
    visited[v] = 1;
    for (int i = 0; i < V; i++) {
        if (graph[v][i] != INF && !visited[i]) {
            topologicalSort(i, visited, stack, top, graph, V);
        }
    }
    stack[(*top)++] = v;
}

void findShortestPath(int graph[MAX][MAX], int V, int start) {
    int distance[MAX], stack[MAX], top = 0, predecessor[MAX];
    int visited[MAX] = {0};

    for (int i = 0; i < V; i++) {
        distance[i] = INF;
        predecessor[i] = -1;
    }
    distance[start] = 0;

    for (int i = 0; i < V; i++) {
        if (!visited[i]) {
            topologicalSort(i, visited, stack, &top, graph, V);
        }
    }

    while (top > 0) {
        int u = stack[--top];
        if (distance[u] != INF) {
            for (int v = 0; v < V; v++) {
                if (graph[u][v] != INF && distance[u] + graph[u][v] < distance[v]) {
                    distance[v] = distance[u] + graph[u][v];
                    predecessor[v] = u;
                }
            }
        }
    }

    printf("Vertex\tDistance from Source\tPath\n");
    for (int i = 0; i < V; i++) {
        if (distance[i] == INF) {
            printf("%d\tINF\t\t\tNo path\n", i);
        } else {
            printf("%d\t%d\t\t\t", i, distance[i]);
            int path[MAX], count = 0, temp = i;
            while (temp != -1) {
                path[count++] = temp;
                temp = predecessor[temp];
            }
            for (int j = count - 1; j >= 0; j--) {
                printf("%d%s", path[j], j > 0 ? " -> " : "");
            }
            printf("\n");
        }
    }
}

int main() {
    int V, start;
    int graph[MAX][MAX];

    printf("Enter the number of vertices: ");
    scanf("%d", &V);

    printf("Enter the adjacency matrix (use %d for INF):\n", INF);
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            scanf("%d", &graph[i][j]);
            if (graph[i][j] == INF) {
                graph[i][j] = INF;
            }
        }
    }

    printf("Enter the starting vertex: ");
    scanf("%d", &start);

    printMatrix(graph, V);
    findShortestPath(graph, V, start);

    return 0;
}

/*
Enter the number of vertices: 4
Enter the adjacency matrix (use 2147483647 for INF):
0 3 2147483647 4
2147483647 0 -2 2147483647
2147483647 2147483647 0 2
2147483647 2147483647 2147483647^@0
Enter the starting vertex: Adjacency Matrix:
0 3 INF 4
INF 0 -2 INF
INF INF 0 2
INF INF INF 0
Vertex  Distance from Source    Path
0       0                       0
1       3                       0 -> 1
2       1                       0 -> 1 -> 2
3       3                       0 -> 1 -> 2 -> 3
*/
