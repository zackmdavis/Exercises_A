#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

int comparator(const void *one, const void *other) {
    return *(int *)one - *(int *)other;
}

bool legal_triangle(int a, int b, int c) {
    int lengths[3] = {a, b, c};
    qsort(lengths, 3, sizeof(int), comparator);
    return lengths[0] + lengths[1] > lengths[2];
}

void the_first_star() {
    FILE *fp;
    int a, b, c;
    int triangles = 0;
    fp = fopen("triangle_specs.txt", "r");
    while (fscanf(fp, "%d %d %d", &a, &b, &c) != EOF) {
        if (legal_triangle(a, b, c)) {
            printf("There exists a triangle with sidelengths %d, %d, and %d!\n",
                   a, b, c);
            triangles++;
        } else {
            printf("It is impossible to conceive of a triangle with sidelengths "
                   "%d, %d, and %d!\n",
                   a, b, c);
        }
    }
    printf("Final triangle representation: %d\n", triangles);
}

void the_second_star() {
    FILE *fp;
    int a1, b1, c1;
    int a2, b2, c2;
    int a3, b3, c3;
    int scan;
    int triangles = 0;
    fp = fopen("triangle_specs.txt", "r");
    while (true) {
        scan = fscanf(fp, "%d %d %d", &a1, &b1, &c1);
        if (scan == EOF) break;
        scan = fscanf(fp, "%d %d %d", &a2, &b2, &c2);
        if (scan == EOF) break;
        scan = fscanf(fp, "%d %d %d", &a3, &b3, &c3);
        if (scan == EOF) break;
        if (legal_triangle(a1, a2, a3)) triangles++;
        if (legal_triangle(b1, b2, b3)) triangles++;
        if (legal_triangle(c1, c2, c3)) triangles++;
    }
    printf("Read-vertically triangle representation: %d\n", triangles);
}

int main() {
    the_first_star();
    the_second_star();
    return 0;
}
