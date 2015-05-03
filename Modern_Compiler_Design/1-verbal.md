**1.1.** *Concerning advantages and disadvantages of implementing a compiler
in the language that it implements.* There's a weak sort of recursive
self-improvement as improvements to the compiler also improve compiler
development, which could be construed as an advantage. Disadvantages
include bootstrapping being a pain (did I ever tell you about the time
I tried to compile Rust, but I didn't have enough memory and it took
like two hours?) and the perils of [trusting
trust](https://www.ece.cmu.edu/~ganger/712.fall02/papers/p761-thompson.pdf):
if you need the old version of Blub to compile the new version of
Blub, mere inspection of the Blub source code can't assure you that
Blub is free of treasonous backdoors.

**1.2.** *Concerning why a frontend would need to know something about
the backend or *vice versa*.* For example, in the [Hy programming
language](http://docs.hylang.org/en/latest/), a Lisp (which I am
[honored to have contributed
to](https://github.com/hylang/hy/commits?author=zackmdavis)) that
compiles to Python's abstract syntax tree and which I would think
doesn't really *have* a backend, the compiler needs to know which
version of Python it's targeting so that it can generate the
appropriate AST (including, say, deciding whether to do cool
iteroperability hacks like [backporting `yield from` to Python
2](http://dustycloud.org/blog/how-hy-backported-yield-from-to-python2/)). In
the other direction, including [debugging
symbols](http://en.wikipedia.org/wiki/Debug_symbol) into a compiled
binary is an important reason a backend might need to know about a
frontend.

**1.4.**
```
          sentence
       /      |    \ 
      /       |     \
 subject    verb     object
    |         |        |
 personal     |      personal
 pronoun      |      pronoun
    |         |        |
    I        see      you
```

__1.7.__ _(On which of the illustrated compiler modules would call
upon on error-reporting module.)_ I feel like several of them could
need to call the error-reporinting module depending on where the error
occurred, but if we imagine that a valid AST produces valid code, then
we would imagine errors happening before that, the lexer and parser
raising errors about bad tokens and bad syntax, respectively.

__1.10__ I think "our" demo compiler (scare quotes because my
attempted Python is a very loose translation) can be said to be
_broad_ rather than _narrow_. `Parse_program` deals with the whole
program.

**1.13.** *Concerning the term "extended subset."* If Bar is an
extended subset of Quux, then Bar doesn't do everything Quux does
("subset"), and it also does things that Quux does not do
("extended"). This being the case, why even bother pretending to be an
implementation of Quux?  If you can't pay the price, you should do the
honest thing and settle for being *Quux-like*.

