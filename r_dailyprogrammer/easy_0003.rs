// http://www.reddit.com/r/dailyprogrammer/comments/pkw2m/2112012_challenge_3_easy/

// The prompt is "Ceasar cipher"

use std::old_io;

fn shift_letters(shift: usize) -> Vec<&str> {
     // XXX: it is written that "this function's return type contains a
     // borrowed value, but the signature does not say which one of
     // `shift`'s 0 elided lifetimes it is borrowed from"
    let alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                    "U", "V", "W", "X", "Y", "Z"];
    let end_slice = &alphabet[(26-shift)..26];
    let start_slice = &alphabet[0..(26-shift)];
    let mut shifted = Vec::new();
    for letter in end_slice.iter() {
        shifted.push(letter);
    }
    for letter in start_slice.iter() {
        shifted.push(letter);
    }
    shifted
}

fn main() {
    println!("Ceasar cipher! How much should we shift?");
    let input = old_io::stdin().read_line().ok()
        .expect("IO failure message here");
    let shift: Option<usize> = input.parse();
}

#[test]
fn test_shift_letters() {
    let shifted_four = vec!["W", "X", "Y", "Z", "A", "B", "C", "D", "E",
                            "F", "G", "H", "I", "J", "K", "L", "M", "N",
                            "O", "P", "Q", "R", "S", "T", "U", "V"];
    assert_eq!(shifted_four, shift_letters(4));
}
