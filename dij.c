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

// Function to print the shortest path from source to vertex v
void printPath(int parent[MAX], int v) {
    if (v == -1) {
        return;
    }
    printPath(parent, parent[v]);
    printf("%d ", v);
}

void dijkstra(int V, int matrix[MAX][MAX], int source) {
    int dist[V], parent[V];
    int visited[V];
    
    // Initialize all distances as INF, visited as false, and parent as -1
    for (int i = 0; i < V; i++) {
        dist[i] = INF;
        visited[i] = 0;
        parent[i] = -1;
    }

    dist[source] = 0;

    for (int count = 0; count < V - 1; count++) {
        // Find the minimum distance vertex
        int min = INF, u;
        for (int i = 0; i < V; i++) {
            if (!visited[i] && dist[i] < min) {
                min = dist[i];
                u = i;
            }
        }
        visited[u] = 1;

        // Update distance of adjacent vertices of the picked vertex
        for (int v = 0; v < V; v++) {
            if (!visited[v] && matrix[u][v] != INF && dist[u] + matrix[u][v] < dist[v]) {
                dist[v] = dist[u] + matrix[u][v];
                parent[v] = u;
            }
        }
    }

    // Print the results
    printf("Vertex\tDistance\tPath\n");
    for (int i = 0; i < V; i++) {
        if (dist[i] == INF) {
            printf("%d\tINF\t\tNo path\n", i);
        } else {
            printf("%d\t%d\t\t", i, dist[i]);
            printPath(parent, i);
            printf("\n");
        }
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

    int source;
    printf("Enter the source vertex: ");
    scanf("%d", &source);

    dijkstra(V, matrix, source);

    return 0;
}
/*
Enter the number of vertices: 6
Enter the adjacency matrix (use 2147483647 for INF):
0 2 4 2147483647 2147483647 2147483647
2147483647 0 1 4 2 2147483647
2147483647 2147483647 0 2147483647 3 2147483647
2147483647 2147483647 2147483647 0 2147483647 2
2147483647 2147483647 2147483647 3 0 2
2147483647 2147483647 2147483647 2147483647 2147483647 0
Adjacency Matrix:
0 2 4 INF INF INF
INF 0 1 4 2 INF
INF INF 0 INF 3 INF
INF INF INF 0 INF 2
INF INF INF 3 0 2
INF INF INF INF INF 0
Enter the source vertex: 0
Vertex  Distance        Path
0       0               0
1       2               0 1
2       3               0 1 2
3       6               0 1 3
4       4               0 1 4
5       6               0 1 4 5
*/
