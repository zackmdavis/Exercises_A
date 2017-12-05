#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>

#include <openssl/md5.h>

bool interesting(char *hash) {
    if (strncmp("00000", hash, 5) == 0) {
        return true;
    } else {
        return false;
    }
}

void first_star_selection_method(char *digest, char *password, int *counter) {
    int next_password_char;
    for (int i = 0; i < 8; i++) {
        if (password[i] == 0) {
            next_password_char = i;
            break;
        }
    }
    password[next_password_char] = digest[5];
    (*counter)++;
}

void second_star_selection_method(char *digest, char *password, int *counter) {
    if (!isdigit(digest[5])) {
        return;
    }
    int position = digest[5] - '0';
    if (position >= 8 || password[position] != 0) {
        return;
    } else {
        printf("setting password[%d] to '%c' and incrementing counter to %d\n",
               position, digest[6], *counter+1);
        password[position] = digest[6];
        (*counter)++;
    }
}

char *hash_search(char *door_id, void (*selection)(char*, char*, int*)) {
    int i = 0;
    int pass_chars = 0;
    char *password = (char *)malloc(8 * sizeof(char));
    memset(password, 0, 8);
    char preimage[40];
    char digest[32];
    while (pass_chars < 8) {
        if (i % 1000000 == 0) {
            printf("still working; i==%d\n", i);
        }
        sprintf(preimage, "%s%d", door_id, i);
        unsigned char *hash = MD5(preimage, strlen(preimage), NULL);
        for (int j = 0; j < 16; j++) {
            sprintf(digest+(j*2), "%02x", (unsigned int)hash[j]);
        }

        if (interesting(digest)) {
            printf("interesting digest %s found at hashdex %d\n", digest, i);
            selection(digest, password, &pass_chars);
        }
        i++;
    }
    return password;
}

int main() {
    printf("%s\n", hash_search("abc", first_star_selection_method));
    printf("%s\n", hash_search("uqwqemis", first_star_selection_method));
    printf("%s\n", hash_search("abc", second_star_selection_method));
    printf("%s\n", hash_search("uqwqemis", second_star_selection_method));
}
