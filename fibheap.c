#include <stdio.h>
#include <stdlib.h>

typedef struct node {
    struct node* parent;
    struct node* child;
    struct node* left;
    struct node* right;
    int key;
    int degree;
} Node;

Node* mini = NULL;

void insertion(int val) {
    Node* new_node = (Node*)malloc(sizeof(Node));
    if (new_node == NULL) {
        printf("Memory allocation failed\n");
        return;
    }
    new_node->key = val;
    new_node->parent = NULL;
    new_node->child = NULL;
    new_node->left = new_node;
    new_node->right = new_node;
    new_node->degree = 0;

    if (mini != NULL) {
        mini->left->right = new_node;
        new_node->right = mini;
        new_node->left = mini->left;
        mini->left = new_node;
        if (new_node->key < mini->key) {
            mini = new_node;
        }
    } else {
        mini = new_node;
    }
    
    printf("Inserted node with key %d\n", val);
}

void display(Node* mini) {
    if (mini == NULL) {
        printf("The Heap is Empty\n");
        return;
    }

    Node* ptr = mini;
    printf("The root nodes of Heap are:\n");
    do {
        printf("%d", ptr->key);
        ptr = ptr->right;
        if (ptr != mini) {
            printf(" --> ");
        }
    } while (ptr != mini);
    printf("\n");
}

void find_min(Node* mini) {
    if (mini != NULL) {
        printf("\nMin of heap is: %d\n", mini->key);
    } else {
        printf("Heap is empty.\n");
    }
}

int main() {
    int n, i, e;
    printf("Enter the number of insertions: ");
    scanf("%d", &n);
    for (i = 1; i <= n; i++) {
        printf("Enter the %d node to insert: ", i);
        scanf("%d", &e);
        insertion(e);
    }

    display(mini);
    find_min(mini);

    return 0;
}
/*
Enter the number of insertions: 12
Enter the 1 node to insert: 2
Inserted node with key 2
Enter the 2 node to insert: 3
Inserted node with key 3
Enter the 3 node to insert: 4
Inserted node with key 4
Enter the 4 node to insert: 6
Inserted node with key 6
Enter the 5 node to insert: 7
Inserted node with key 7
Enter the 6 node to insert: 3
Inserted node with key 3
Enter the 7 node to insert:
6
Inserted node with key 6
Enter the 8 node to insert: 7
Inserted node with key 7
Enter the 9 node to insert: 5
Inserted node with key 5
Enter the 10 node to insert: 2
Inserted node with key 2
Enter the 11 node to insert: 4
Inserted node with key 4
Enter the 12 node to insert: 5
Inserted node with key 5
The root nodes of Heap are:
2 --> 3 --> 4 --> 6 --> 7 --> 3 --> 6 --> 7 --> 5 --> 2 --> 4 --> 5

Min of heap is: 2
*/
