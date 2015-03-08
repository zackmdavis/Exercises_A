// "Implement an algorithm to determine if a string has all unique characters."

use std::collections::HashSet;

fn letter_unique(word: &'static str) -> bool {
    let mut seen_that_already = HashSet::new();
    for c in word.chars() {
        if seen_that_already.contains(&c) {
            return false
        }
        seen_that_already.insert(c);
    }
    true
}

// Imitating another approach from the text's solutions section

// XXX I haven't compiled this yet
fn is_unique_chars(our_string: &str) -> bool {
    let mut char_set: [bool; 256] = [false; 256];
    for c in our_string.chars() {
        let ord: u32 = c as u32;
        if char_set[ord] {
            return false;
        }
        char_set[ord] = true;
    }
}

#[test]
fn test_letter_uniqueness() {
    assert!(letter_unique("ambidextrously"));  // OK
    assert!(letter_unique("uncopywritable"));  // OK
    assert!(!letter_unique("glitter"));  // two 't's
    assert!(!letter_unique("senselessness"));  // six 's's, four 'e's, two 'n's
}

#[test]
fn test_is_unique_chars() {
    // TODO
}
