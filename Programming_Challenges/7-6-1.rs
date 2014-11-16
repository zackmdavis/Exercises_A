
fn on(n: uint) -> bool {
    let mut lights = Vec::from_elem(n, false);
    for pass in range(1u, n + 1) {
        for i in range(1, n + 1) {
            let flip_it = i % pass == 0;
            if flip_it {
                *lights.get_mut(i - 1) = !lights.get(i - 1);
            }
        }
    }
    *lights.get(n - 1)
}

fn perfect_square(n: uint) -> bool {
    let nternal = n as f64;
    let square_root = nternal.sqrt();
    let nearest_integer = square_root.floor();
    square_root == nearest_integer
}

#[test]
fn test_perfect_square() {
    assert!(perfect_square(4));
    assert!(perfect_square(36));
    assert!(!perfect_square(37));
}

#[test]
fn test_sample_output() {
    for i in range(5u, 100) {
        assert!(perfect_square(i) == on(i));
    }
}
