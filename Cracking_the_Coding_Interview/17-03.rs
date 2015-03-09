// "Write an algorithm which computes the number of trailing zeros in
// n factorial"

// That's easy; just count the factors of two and five.

use std::cmp::min;


fn primes() -> Vec<u32> {
    vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
         67, 71, 73, 79, 83, 89, 97, 101, 103]
        // That's probably enough for now.
}

fn trial_division_factorize(n: u32) -> Vec<u32> {
    let mut factors: Vec<u32> = Vec::new();
    let mut presently: u32 = n;
    while presently != 1 {
        for p in primes() {
            if presently % p == 0 {
                factors.push(p);
                presently /= p;
                break;
            }
        }
    }
    factors
}

fn factorial(n: u32) -> u32 {
    if n == 0 || n == 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

fn trailing_zeros_in_factorial(n: u32) -> u32 {
    let mut factors_of_two: u32 = 0;
    let mut factors_of_five: u32 = 0;
    for i in 2..(n+1) {
        let more_factors = trial_division_factorize(i);
        for p in more_factors {
            match p {
                2 => { factors_of_two += 1; },
                5 => { factors_of_five += 1; },
                _ => ()
            }
        }
    }
    min(factors_of_two, factors_of_five)
}

#[test]
fn test_trial_division_factorize() {
    assert_eq!(vec![2, 3, 5], trial_division_factorize(30));
    assert_eq!(vec![2, 2, 2], trial_division_factorize(8));
    assert_eq!(vec![3, 5, 11, 41], trial_division_factorize(6765));
}

#[test]
fn test_factorial() {
    assert_eq!(factorial(3), 6);
    assert_eq!(factorial(5), 120);
    assert_eq!(factorial(11), 39916800);
}

#[test]
fn test_trailing_zeros_in_factorial() {
    assert_eq!(trailing_zeros_in_factorial(7), 1);  // 5040
    assert_eq!(trailing_zeros_in_factorial(11), 2);  // 39916800
    assert_eq!(trailing_zeros_in_factorial(20), 4);  // 2432902008176640000
    assert_eq!(trailing_zeros_in_factorial(100), 24);  // &c.
}
