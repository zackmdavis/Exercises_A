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

#[test]
fn test_letter_uniqueness() {
    assert!(letter_unique("ambidextrously"));  // OK
    assert!(letter_unique("uncopywritable"));  // OK
    assert!(!letter_unique("glitter"));  // two 't's
    assert!(!letter_unique("senselessness"));  // six 's's, four 'e's, two 'n's
}


// XXX BROKEN RETARDED: the commented-out (and really, what kind of
// jerk checks commented-out code into version control?!) code below is
// confused and does not work and is almost cetainly nonsense because
// I still don't understand Rust's type system

// enum Stringlike {
//     S1(&'static str),
//     S2(String)
// }

// fn letter_unique_redux(word: Stringlike) -> bool {
//     let mut seen_that_already = HashSet::new();
//     for c in word.chars() {
//         if seen_that_already.contains(&c) {
//             return false
//         }
//         seen_that_already.insert(c);
//     }
//     true
// }

// fn test_type_polymorphic_letter_uniqueness() {
//     assert!(letter_unique_redux(Stringlike("uncopywritable")));  // OK
//     assert!(!letter_unique_redux(Stringlike("glitter")));  // two 't's

//     assert!(letter_unique_redux(Stringlike("dog").to_string()));  // OK
//     assert!(!letter_unique_redux(
//         Stringlike("inconveniencing").to_string()));  // 3 'i's, &c.
// }
