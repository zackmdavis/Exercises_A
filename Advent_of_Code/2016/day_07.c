#include <stdbool.h>
#include <string.h>
#include <stdio.h>


bool has_autonomous_bridge_bypass_annotation(char *address) {
    int hypernet_counter = 0;
    bool found_annotation = false;
    for (int i = 0; i < strlen(address) - 4; i++) {
        if (address[i] == '[') {
            hypernet_counter++;
        } else if (address[i] == ']') {
            hypernet_counter--;
        }

        if (address[i] == address[i+3] &&
            address[i+1] == address[i+2] &&
            address[i] != address[i+1]) {
            if (hypernet_counter == 0) {
                found_annotation = true;
            } else {
                return false;
            }
        }
    }
    return found_annotation;
}

struct SequenceOfInterest {
    char first;
    char second;
};


bool has_super_secret_listening(char *address) {
    int hypernet_counter = 0;
    struct SequenceOfInterest area_broadcast_accessors[256];
    struct SequenceOfInterest byte_allocation_blocks[256];
    int abas = 0;
    int babs = 0;
    for (int i = 0; i < strlen(address) - 3; i++) {
        if (address[i] == '[') {
            hypernet_counter++;
        } else if (address[i] == ']') {
            hypernet_counter--;
        }

        if (address[i] == address[i+2] && address[i] != address[i+1]) {
            if (hypernet_counter == 0) {
                area_broadcast_accessors[abas] = (struct SequenceOfInterest) {
                    address[i], address[i+1]
                };
                abas++;
            } else {
                byte_allocation_blocks[babs] = (struct SequenceOfInterest) {
                    address[i], address[i+1]
                };
                babs++;
            }
        }
    }

    for (int a = 0; a < abas; a++) {
        for (int b = 0; b < babs; b++) {
            if (area_broadcast_accessors[a].first == byte_allocation_blocks[b].second &&
                area_broadcast_accessors[a].second == byte_allocation_blocks[b].first) {
                printf("%c %c %c %c %s\n", area_broadcast_accessors[a].first, byte_allocation_blocks[b].second, area_broadcast_accessors[a].second, byte_allocation_blocks[b].first, address);
                return true;
            }

        }
    }
    return false;
}

int main() {
    FILE *fp = fopen("input.txt", "r");
    if (fp == NULL) {
        printf("no such file, dummy\n");
        return 1;
    }
    char buf[256];
    int first_star_counter = 0;
    int second_star_counter = 0;
    while (fgets(buf, 200, fp) != NULL) {
        if (has_autonomous_bridge_bypass_annotation(buf)) {
            first_star_counter++;
        }
        if (has_super_secret_listening(buf)) {

            second_star_counter++;
        }
    }
    printf("%d\n%d\n", first_star_counter, second_star_counter);
}
