#include <stdio.h>
#include <stdlib.h>

void increment(int counter[], int n, int *c) {
    int i = 0;
    while ((i < n + 1) && (counter[i] == 1)) {
        counter[i] = 0;
        i++;
        (*c)++;
    }

    if (i < n + 1) {
        counter[i] = 1;
        (*c)++;
    }
}

void printCounter(int counter[], int n, int y) {
    printf("%d = ", y); // y -> current count or current iteration value
    for (int i = n; i >= 0; i--) {
        printf("%d ", counter[i]);
    }
    printf("\n");
}

int main() {
    int counter[10] = {0};
    int y = 0, c = 0;
    int k, n;

    printf("Enter the limit of the counter: ");
    scanf("%d", &k);

    n = 4;

    printCounter(counter, n, y);

    while (y < k) {
        increment(counter, n, &c); // c -> pointer to the operation counter
        y++;
        printCounter(counter, n, y);
    }

    printf("Amortized cost of %d operations is %.1f\n", k, (float)c);
    printf("Average cost is %d/%d = %.2f\n", c, k, (float)c / k);

    return 0;
}
*/
Enter the limit of the counter: 16
0 = 0 0 0 0 0
1 = 0 0 0 0 1
2 = 0 0 0 1 0
3 = 0 0 0 1 1
4 = 0 0 1 0 0
5 = 0 0 1 0 1
6 = 0 0 1 1 0
7 = 0 0 1 1 1
8 = 0 1 0 0 0
9 = 0 1 0 0 1
10 = 0 1 0 1 0
11 = 0 1 0 1 1
12 = 0 1 1 0 0
13 = 0 1 1 0 1
14 = 0 1 1 1 0
15 = 0 1 1 1 1
16 = 1 0 0 0 0
Amortized cost of 16 operations is 31.0
Average cost is 31/16 = 1.94
*/
