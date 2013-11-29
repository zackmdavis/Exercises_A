#include <stdio.h>

int main(int argc, char *argv[])
{
  int areas[] = {10, 23, 23, 12, 32};
  char name[] = "Zack";
  char full_name[] = {
    'Z', 'a', 'c', 'k', ' ',
    'M', '.', ' ',
    'D', 'a', 'v', 'i', 's', '\0', 
  };

  printf("The size of an int: %ld\n", sizeof(int));
  printf("The size of areas (int[]): %ld\n", sizeof(areas));
  printf("The number of ints in areas: %ld\n", sizeof(areas) / sizeof(int));
  printf("The size of name: %ld\n", sizeof(name));
  printf("full_name=\"%s\"\n", full_name);
  return 0;
}
