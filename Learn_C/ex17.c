#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#define MAX_DATA 512
#define MAX_ROWS 100

struct Address {
  int id;
  int set;
  char name[MAX_DATA];
  char email[MAX_DATA];
};

struct Database {
  struct Address rows[MAX_ROWS];
};

struct Connection {
  FILE *file;
  struct Database *db;
};

void die(const char *message)
{
  if (errno) {
    perror(message);
  } else {
    printf("ERROR: %s\n", message);
  }
  exit(1);
}

void Address_print(struct Address *address)
{
  printf("%d %s %s\n", address->id, address->name, address->email);
}

void Database_load(struct Connection *cxn)
{
  int rc = fread(cxn->db, sizeof(struct Database), 1, cxn->file);
  if (rc != 1) {
    die("Failed to load database.");
  }
}

struct Connection *Database_open(const char *filename, char mode)
{
  struct Connection *cxn = malloc(sizeof(struct Connection));
  if (!cxn) {
    die("Memory error");
  }

  cxn->db = malloc(sizeof(struct Database));
  if (!cxn->db) {
    die("Memory error");
  }

  if (mode == 'c') {
    cxn->file = fopen(filename, "w");
  } else { 
    cxn->file = fopen(filename, "r+");
    
    if (cxn->file) {
      Database_load(cxn);
    }
  }

  if (!cxn->file) {
    die("Failed to open the file");
  }

  return cxn;
}

void Database_close(struct Connection *cxn)
{
  if (cxn) {
    if (cxn->file) fclose(cxn->file);
    if (cxn->db) free(cxn->db);
    free(cxn);
  }
}

void Database_write(struct Connection *cxn)
{
  rewind(cxn->file);

  int rc = fwrite(cxn->db, sizeof(struct Database), 1, cxn->file);
  if (rc != 1) die("Failed to write database");

  rc = fflush(cxn->file);
  if (rc == -1) die ("Cannot flush database.");
}

void Database_create(struct Connection *cxn)
{
  for (int i = 0; i < MAX_ROWS; i++) {
    struct Address address = {.id = i, .set = 0};
    cxn->db->rows[i] = address;
  }
}

void Database_set(struct Connection *cxn, int id, const char *name, const char *email)
{
  struct Address *address = &cxn->db->rows[id];
  if (address->set) die("Already set, delete it first");

  address->set = 1;
  char *result = strncpy(address->name, name, MAX_DATA);
  if (!result) die("Name copy failed");

  result = strncpy(address->email, email, MAX_DATA);
  if (!result) die("Email copy failed");
}

void Database_get(struct Connection *cxn, int id)
{
  struct Address *address = &cxn->db->rows[id];

  if(address->set) {
    Address_print(address);
  } else {
    die("ID is not set");
  }
}

void Database_delete(struct Connection *cxn, int id)
{
  struct Address address = {.id = id, .set = 0};
  cxn->db->rows[id] = address;
}

void Database_list(struct Connection *cxn)
{
  struct Database *db = cxn->db;

  for(int i = 0; i < MAX_ROWS; i++) {
    struct Address *current = &db->rows[i];

    if (current->set) {
      Address_print(current);
    }
  }
}

int main(int argc, char *argv[])
{
  if (argc < 3) die("USAGE: ex17.o <dbfile> <action> [action parameters]");

  char *filename = argv[1];
  char action = argv[2][0];
  struct Connection *cxn = Database_open(filename, action);
  int id = 0;

  if (argc > 3) id = atoi(argv[3]);
  if (id >= MAX_ROWS) die("There aren't that many records");

  switch(action) {
  case 'c':
    Database_create(cxn);
    Database_write(cxn);
    break;

  case 'g':
    if (argc != 4) die("Need an ID");

    Database_get(cxn, id);
    break;

  case 's':
    if (argc != 6) die("Need ID, name, email to set");

    Database_set(cxn, id, argv[4], argv[5]);
    Database_write(cxn);
    break;

  case 'd':
    if (argc != 4) die("Need id to delete");

    Database_delete(cxn, id);
    Database_write(cxn);
    break;

  case 'l':
    Database_list(cxn);
    break;

  default:
    die("Invalid action; only _c_reate _g_et _s_et _d_elete or _l_ist"); 
  }
  Database_close(cxn);

  return 0;
}
