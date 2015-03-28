// http://www.reddit.com/r/dailyprogrammer/comments/pkw2m/2112012_challenge_3_easy/

// The prompt is "Ceasar cipher"

use std::old_io;

fn shift_letters(shift: usize) -> Vec<&str> {
    // XXX THE LANGUAGE IS A CRUEL HOAX: "easy_0003.rs:7:39: 7:43 help:
    // this function's return type contains a borrowed value, but the
    // signature does not say which one of `shift`'s 0 elided lifetimes
    // it is borrowed from"---now, I may be but a humble Pythonista from
    // the country, but I'm pretty sure you can't talk about "which one"
    // of a set of size zero
    let mut start_slice = vec!["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                               "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                               "U", "V", "W", "X", "Y", "Z"];
    // Wait for it ...
    let end_slice = start_slice.split_off();
    let mut shifted = Vec::new();
    for letter in end_slice {
        shifted.push(letter);
    }
    for letter in start_slice {
        shifted.push(letter)
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
