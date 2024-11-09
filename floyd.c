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

void floydWarshall(int V, int matrix[MAX][MAX]) {
    int distance[MAX][MAX];
    int cycleCount = 0;

    // Initialize distance matrix
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            distance[i][j] = matrix[i][j];
        }
    }

    // Floyd-Warshall algorithm to find the shortest path
    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (distance[i][k] != INF && distance[k][j] != INF &&
                    distance[i][k] + distance[k][j] < distance[i][j]) {
                    distance[i][j] = distance[i][k] + distance[k][j];
                }
            }
        }
    }

    // Check for negative-weight cycles
    for (int i = 0; i < V; i++) {
        if (distance[i][i] < 0) {
            cycleCount++;
        }
    }

    if (cycleCount > 0) {
        printf("Number of negative cycles detected: %d\n", cycleCount);
    } else {
        printf("No negative cycles detected.\n");
    }

    printf("Shortest Path Matrix:\n");
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (distance[i][j] == INF) {
                printf("INF ");
            } else {
                printf("%d ", distance[i][j]);
            }
        }
        printf("\n");
    }
}

int main() {
    int V;
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

    printMatrix(matrix, V);  // Print the adjacency matrix
    floydWarshall(V, matrix); // Run Floyd-Warshall and detect cycles

    return 0;
}

/*
Enter the number of vertices: 5
Enter the adjacency matrix (use 2147483647 for INF):
0 3 8 2147483647 -4
2147483647 0 2147483647 1 7
2147483647 4 0 2147483647 2147483647
2 2147483647 -5 0 2147483647
2147483647 2147483647 2147483647 6 0
Adjacency Matrix:
0 3 8 INF -4
INF 0 INF 1 7
INF 4 0 INF INF
2 INF -5 0 INF
INF INF INF 6 0
No negative cycles detected.
Shortest Path Matrix:
0 1 -3 2 -4
3 0 -4 1 -1
7 4 0 5 3
2 -1 -5 0 -2
8 5 1 6 0
*/
