// "Given two strings, write a method to decide if one is a
// permutation of the other."

// Okay, well, let's count how many times each letter appears; the
// total counts will be the same iff the strings are permutations of each
// other.

use std::collections::hashmap::HashMap;

fn count_letters(word: String) -> HashMap<char, uint> {
    let mut counter = HashMap::new();
    for letter in word.as_slice().chars() {
        if counter.contains_key(&letter) {
            let previous_count: uint = *counter.get(&letter);
            counter.insert(letter, previous_count + 1u);
        } else {
            counter.insert(letter, 1u);
        }
    }
    counter
}

fn counters_equal(first: HashMap<char, uint>,
                  second: HashMap<char, uint>) -> bool {
    for (key, value) in first.iter() {
        if second.get(key) != value {
            return false
        }
    }
    // but what if `second` has extra keys? better do the same thing
    // with roles reversed (reversed roles?)
    //
    // (yes, this is a really really stupid algorithm
    // and not knowing how to do anything in Rust is no excuse)
    for (key, value) in second.iter() {
        if first.get(key) != value {
            return false
        }
    }
    true
}

fn permuted(first: String, second: String) -> bool {
    counters_equal(count_letters(first), count_letters(second))
}

#[test]
fn test_counters_equal() {
    let mut test_counter_1 = HashMap::new();
    test_counter_1.insert(97 as char, 1u);
    test_counter_1.insert(98 as char, 2u);
    test_counter_1.insert(99 as char, 3u);
    let mut test_counter_2 = HashMap::new();
    test_counter_2.insert(99 as char, 3u);
    test_counter_2.insert(98 as char, 2u);
    test_counter_2.insert(97 as char, 1u);
    assert!(counters_equal(test_counter_1, test_counter_2));
}

#[test]
fn test_permuted() {
    assert!(permuted("rusty".to_string(), "yurts".to_string()));
}
