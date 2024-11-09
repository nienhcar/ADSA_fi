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

void bellmanFord(int V, int matrix[MAX][MAX], int start) {
    int distance[MAX];
    int predecessor[MAX];
    int cycleCount = 0;

    // Initialize distances and predecessors
    for (int i = 0; i < V; i++) {
        distance[i] = INF;
        predecessor[i] = -1;
    }
    distance[start] = 0;

    // Relax all edges |V|-1 times
    for (int i = 1; i <= V - 1; i++) {
        for (int u = 0; u < V; u++) {
            for (int v = 0; v < V; v++) {
                if (matrix[u][v] != INF && distance[u] != INF && distance[u] + matrix[u][v] < distance[v]) {
                    distance[v] = distance[u] + matrix[u][v];
                    predecessor[v] = u;
                }
            }
        }
    }

    // Check for negative-weight cycles
    for (int u = 0; u < V; u++) {
        for (int v = 0; v < V; v++) {
            if (matrix[u][v] != INF && distance[u] != INF && distance[u] + matrix[u][v] < distance[v]) {
                cycleCount++;
                break;
            }
        }
    }

    if (cycleCount > 0) {
        printf("Number of negative cycles detected: %d\n", cycleCount);
    } else {
        printf("No negative cycles detected.\n");
        printf("Vertex\tDistance from Source\n");
        for (int i = 0; i < V; i++) {
            printf("%d\t\t%d\n", i, distance[i] == INF ? -1 : distance[i]);
        }
    }
}

int main() {
    int V, start;
    int matrix[MAX][MAX];

    printf("Enter the number of vertices: ");
    scanf("%d", &V);

    printf("Enter the adjacency matrix (use %d for INF):\n", INF);
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            scanf("%d", &matrix[i][j]);
            if (matrix[i][j] == INF) {
                matrix[i][j] = INF;
            }
        }
    }

    printf("Enter the starting vertex: ");
    scanf("%d", &start);

    printMatrix(matrix, V);  // Print the adjacency matrix
    bellmanFord(V, matrix, start);

    return 0;
}
/*
 rachana@LAPTOP-FPSS0O9R:/mnt/c/Users/racha/Desktop/ccodes/ADSA$ ./bellman
Enter the number of vertices: 4
Enter the adjacency matrix (use 2147483647 for INF):
0 1 2147483647 4
2147483647 0 -2 2147483647
2147483647 2147483647 0 3
-5 2147483647 2147483647 0
Enter the starting vertex: 0
Adjacency Matrix:
0 1 INF 4
INF 0 -2 INF
INF INF 0 3
-5 INF INF 0
Number of negative cycles detected: 1
*/
