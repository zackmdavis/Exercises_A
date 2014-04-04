#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "object.h"
#include <assert.h>

void Object_destroy(void *self)
{
  Object *obj = self;
  if (obj) {
    if (obj->description) {
      free(obj->description);
    }
    free(obj);
  }
}

void Object_describe(void *self)
{
  Object *obj = self;
  printf("%s.\n", obj->description);
}

int Object_init(void *self)
{
  // this is kind of a fake function
  return 1;
}

void *Object_move(void *self, Direction direction)
{
  printf("You can't go in that direction.\n");
  return NULL;
}

int Object_attack(void *self, int damage)
{
  printf("You can't attack that.\n");
}

void *Object_new(size_t size, Object proto, char *description)
{
  /* set default functions if not already set (and it's funny,
     but I had almost thought that truthiness was a modern
     dynamic-language thing) */
  if (!proto.init) {
    proto.init = Object_init;
  }
  if (!proto.describe) {
    proto.describe = Object_describe;
  }
  // (the following conditionals brought to you by the power of keyboard macros)
  if (!proto.destroy) {
    proto.destroy = Object_destroy;
  }
  if (!proto.attack) {
    proto.attack = Object_attack;
  }
  if (!proto.move) {
    proto.move = Object_move;
  }

  Object *el = calloc(1, size);
  *el = proto;

  // copy description
  el->description = strdup(description);

  if(!el->init(el)) {
    // if it didn't initialize right... trash it?
    el->destroy(el);
    return NULL;
  } else {
    // all done, we made an object of any type
    return el;
  }
}
