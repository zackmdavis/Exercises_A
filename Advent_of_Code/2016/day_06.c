#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// TODO: debug

char most_frequent_letter(char letters[]) {
    printf("most_frequent_letters called on %s\n", letters);
    int histogram[26];
    memset(histogram, 0, 26);
    int i = 0;
    int letter;
    do {
        letter = letters[i];
        histogram[letter - 'a']++;
        i++;
    } while (letters[i] != 0);
    int max = -1;
    int argmax = -1;
    for (int j = 0; j < 26; j++) {
        if (histogram[j] > max) {
            max = histogram[j];
            argmax = j;
        }
    }
    return argmax + 'a';
}

char *intercept_transmission_and_decode() {
    FILE *fp = fopen("error_correction.txt", "r");
    char *final_message = (char *)malloc(8 * sizeof(char));

    // first count the number of lines so we know how much to allocate, but
    // we'll assume that we know the message length is 8
    int message_length = 8;
    int copies = 0;
    char buf[10];
    while (fgets(buf, 10, fp) != NULL) {
        copies++;
    }

    char slots[message_length][copies+1];
    fseek(fp, 0, SEEK_SET);
    int copy = 0;
    while (fgets(buf, 10, fp) != NULL) {
        for (int i = 0; i < 8; i++) {
            slots[i][copy] = buf[i];
        }
        copy++;
    }

    for (int c = 0; c < 8; c++) {
        slots[c][copies] = 0;

        /* printf("%c", most_frequent_letter(slots[c])); */
    }
    printf("\n");
}

int main() {
    intercept_transmission_and_decode();
}
