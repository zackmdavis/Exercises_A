// I just got accepted to the GitHub Copilot Technical Preview,
// which would be exciting if it weren't for humanity's existential
// risk situation.
//
// Let's try it out on exercises from /r/dailyprogrammer.

// Warmup: given a number n, determine the number of times the
// digit "1" appears if you write out all numbers from 1 to n
// inclusive.

fn count_ones_in_predecessors(n: u64) {
    // Copilot wrote the body of the function! Very impressive :(
    let mut count = 0;
    for i in 1..n+1 {
        let mut i = i;
        while i > 0 {
            if i % 10 == 1 {
                count += 1;
            }
            i /= 10;
        }
    }
    println!("{}", count);
}

fn main() {}
