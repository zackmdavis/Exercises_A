#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

void die(const char *message)
{
  if (errno) {
    perror(message);
  } else {
    printf ("ERROR: %s\n", message);
  }

  exit(1);
}

typedef int (*compare_cb)(int a, int b);

// TODO: try writing something less dross, like quicksort
int *bubble_sort(int *numbers, int count, compare_cb cmp)
{
  int temp = 0;
  int i = 0;
  int j = 0;
  int *target = malloc(count * sizeof(int));

  if (!target) die("Memory error.");

  memcpy(target, numbers, count * sizeof(int));

  for (i = 0; i < count; i++) {
    for (j = 0; j < count-1; j++) {
      if (cmp(target[j], target[j+1]) > 0) {
	temp = target[j+1];
	target[j+1] = target[j];
	target[j] = temp;
      }
    }
  }
  return target;
}

int ascending_order(int a, int b)
{
  return a - b;
}

int descending_order(int a, int b)
{
  return b - a;
}

int magic_order(int a, int b)
{
  if (a == 0 || b == 0) {
    return 0;
  } else {
    return a % b;
  }
}

void test_sorting(int *numbers, int count, compare_cb cmp)
{
  int i = 0;
  int *sorted = bubble_sort(numbers, count, cmp);

  if(!sorted) die ("Fail");

  for (i = 0; i < count; i++) {
    printf("%d ", sorted[i]);
  }
  printf("\n");

  // let's look at the first 25 bytes of the callback (for teh lulz)
  unsigned char *data = (unsigned char *)cmp;
  for (i = 0; i < 25; i++) {
    printf("%02x:", data[i]);
  }
  printf("\n");

  free(sorted);
}

int main(int argc, char *argv[])
{
  if (argc < 2) die("USAGE: ex18.o [numbers to sort, e.g., 2 4 3 0 1 9]");

  int count = argc - 1;
  int i = 0;
  char **inputs = argv + 1;

  int *numbers = malloc(count * sizeof(int));
  if (!numbers) die("Memory error");

  for (i = 0; i < count; i++) {
    numbers[i] = atoi(inputs[i]);
  }

  test_sorting(numbers, count, ascending_order);
  test_sorting(numbers, count, descending_order);
  test_sorting(numbers, count, magic_order);

  free(numbers);

  return 0;
}
