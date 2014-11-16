fn collatz_procedure(n: uint) -> uint {
    if n % 2 == 1 {
        (3 * n) + 1
    } else {
        n / 2
    }
}

fn collatz_sequence(n: uint) -> Vec<uint> {
    let mut our_sequence = Vec::new();
    let mut i = n;
    loop {
        our_sequence.push(i);
        if i == 1 {
            break;
        }
        i = collatz_procedure(i);
    }
    our_sequence
}

fn max_lifetime(lower: uint, upper: uint) -> uint {
    let mut best_lifetime = 0u;
    for i in range(lower, upper) {
        let lifetime = collatz_sequence(i).len();
        if lifetime > best_lifetime {
            best_lifetime = lifetime;
        }
    }
    best_lifetime
}

#[test]
fn test_known_sequence() {
    assert!(collatz_sequence(22) == vec![22, 11, 34, 17, 52, 26, 13, 40,
                                         20, 10, 5, 16, 8, 4, 2, 1])
}

#[test]
fn test_sample_output() {
    assert!(20 == max_lifetime(1, 11));
    assert!(125 == max_lifetime(100, 201));
    assert!(89 == max_lifetime(201, 211));
    assert!(174 == max_lifetime(900, 1001));
}
