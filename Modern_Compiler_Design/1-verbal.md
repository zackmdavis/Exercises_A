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
the backend or vice versa.* For example, in the [Hy programming
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

__1.14.__ _(Concerning the replacement of a production rule
`expression → expression '+' term | expression '-' term | term` with
`expression → expression '+' expression | expression '-' expression |
expression`, resulting in a grammar that produces the same language,
but the rule replacement being purportedly incorrect.)_ I believe the
problem is that the grammar is _ambiguous_. For our purposes as
aspiring compiler writers, a grammar doesn't just define a language in
the sense of a subset of the space of all strings, it generates
strings in a structured way that we want to attach meanings to, and to
do that reliably, we need to be able to uniquely reconstruct our parse
tree from the string, which we can't do with the grammar with the
replacement rule, because there can be multiple parse trees that match
the same input—

```
      E                E     
    / | \            / | \   
   E  +  E   vs.    E  +  E  
  /|\    |          |    /|\ 
 1 + 2   3          1   2 + 3
```

__1.15.__ _(Concerning how we would rewrite the extended
Backus&ndash;Naur form rule `parameter_list → ('IN | 'OUT')?
identifier (',' identifier)*` in regular, non-extended
Backus&ndash;Naur form.)_ This would seem to just be a matter of
rephrasing our convenient repetition operators in terms of raw
disjunctions or recursion, like this—

```
parameter_list → io ids | ids
io → 'IN' | 'OUT'
ids → identifier | identifier ',' ids
```

__1.16.__ This multiprompt is concerned with this grammar—

```
S → A | B | C
A → B | ε
B → x | C y
C → B C S
```

__a.__ _(Concerning which of the nonterminals, are left- or
right-recursive, nullable, or useless.)_ `A` is _nullable_, `S` is
right-recursive by way of `S → C → B C S`; `C` is right-recursive by
way of `C → BCS → BCC`; `B` is left-recursive by way of `B → Cy → B C
S y`. `C` is (somewhat to my surprise, for it took me a while to
notice) _useless_: the only representation of `C` contains yet another
`C`, so you can never finish cashing it out; deriving a single `C`
anywhere in one's use of the grammar condemns us to wander the
recursive halls of eternity.

__b.__ _(Concerning the language produced by this grammar.)_ This
seemed like a hard exercise until I realized that `C` was useless and
must not be touched. That leaves our only alternatives as `A` and `B`;
`A` is either the same as `B` or it is nothing (ε), and `B` is just
the terminal `x`. So the language is that given by the regular
expression `x?`.

__c.__ _(Concerning whether the grammar is ambiguous.)_

Before addressing the content of this exercise prompt, a note about
contemptible metagaming: the text told us that a grammar is ambiguous
if it can produce two production trees with the same leaves in the
same order, so to demonstrate that a grammar is ambiguous, one would
just have to come up with an example of two such trees. But I don't
_remember_ the text explaining any methods to prove that a grammar is
_not_ ambiguous, so sheerly from the standpoint of "we don't expect
textbook authors to ask us questions that they haven't already told us
everything we need to know in order to answer", we already have reason
to expect that the grammar is ambiguous. But this is a scoundrel's
reasoning; _we_ (the exact scope of the plural first-person pronoun is
left undefined, but dear reader, if you're looking at someone else's
exercise solutions for _Modern Compiler Theory_ online for fun, you
are definitely included) are not snivelling schoolstudents trying to
guess the teacher's password; _we_ know that textbooks and the
exercises they contain are merely an aid to help us become stronger,
that we might achieve mastery in our profession and exert dominion
over all of computational existence that we survey. It doesn't matter
whether we can infer the answer the given question from the social
context in which it was asked, exactly because that kind of trick
_doesn't_ work in the real world of computational existence; neither
mathematics nor the market are obliged to present us with solvable
challenges.

Anyway, the grammar is ambiguous. Recall that the language produced by
this grammar only has two members: `x` and the empty string. If our
input is the empty string, well, that can only be matched by `A`, but
if our input is `x`, it's, shall we say, ambiguous whether the parse
tree should contain `A` or not—

```
S — A — B — x

S — B — x
```
