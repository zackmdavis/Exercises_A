// http://www.reddit.com/r/dailyprogrammer/comments/
// 2rnwzf/20150107_challenge_196_intermediate_rail_fence/

// Railfence cipher!

use std::cmp::Ordering;

// Yay, this works! However ...
// TODO: I think we could do this with less boilerplatey or
// copy-pastey code; clean it up as much as possible

#[derive(Debug, PartialEq, Eq, PartialOrd)]
struct RailfenceCipherCell {
    figure: char,
    length: usize,
    depth: usize
}

impl Ord for RailfenceCipherCell {
    fn cmp(&self, other: &RailfenceCipherCell) -> Ordering {
        if self.depth < other.depth {
            Ordering::Less
        } else if self.depth > other.depth {
            Ordering::Greater
        } else {
            if self.length < other.length {
                Ordering::Less
            } else if self.length > other.length {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
    }
}

fn scoot_railfence_counters(zig_depth: usize,
                            length: &mut usize, depth: &mut usize,
                            zig: &mut bool) {
    *length += 1;
    if (*depth == zig_depth - 1) || (*depth == 0) {
        *zig = !*zig;
    }
    if *zig {
        *depth += 1;
    } else {
        *depth -= 1;
    }
}

fn plaintext_to_ciphercells(plaintext: String,
                            zig_depth: usize) -> Vec<RailfenceCipherCell> {
    let mut cells: Vec<RailfenceCipherCell> = Vec::new();
    let mut length: usize = 0;
    let mut depth: usize = 0;
    let mut zig: bool = false;
    for character in plaintext.chars() {
        cells.push(
            RailfenceCipherCell{
                figure: character, length: length, depth: depth }
        );
        scoot_railfence_counters(zig_depth,
                                 &mut length, &mut depth, &mut zig);
    }
    cells
}

fn ciphercells_to_ciphertext(
    ciphercells: &mut Vec<RailfenceCipherCell>) -> String {
    ciphercells.sort();
    let mut ciphertext: String = String::new();
    // having trouble finding the `join` analogue quickly
    for cell in ciphercells.iter() {
        ciphertext.push(cell.figure);
    }
    ciphertext
}

fn ciphercells_to_plaintext(
    ciphercells: &mut Vec<RailfenceCipherCell>) -> String {
    ciphercells.sort_by(|a, b| a.length.cmp(&b.length));
    let mut plaintext: String = String::new();
    for cell in ciphercells.iter() {
        plaintext.push(cell.figure);
    }
    plaintext
}

fn encode(plaintext: String, depth: usize) -> String {
    ciphercells_to_ciphertext(&mut plaintext_to_ciphercells(plaintext, depth))
}


fn ciphertext_to_ciphercells(ciphertext: String, zig_depth: usize) -> Vec<RailfenceCipherCell> {
    let mut addresses: Vec<(usize, usize)> = Vec::new();
    let mut length: usize = 0;
    let mut depth: usize = 0;
    let mut zig: bool = false;
    for _character in ciphertext.chars() {
        addresses.push((depth, length));
        scoot_railfence_counters(zig_depth,
                                 &mut length, &mut depth, &mut zig);
    }
    addresses.sort();
    let mut cells: Vec<RailfenceCipherCell> = Vec::new();
    for (character, &address) in ciphertext.chars().zip(addresses.iter()) {
        cells.push(RailfenceCipherCell{
            depth: address.0, length: address.1, figure: character}
        );
    }
    cells
}

fn decode(ciphertext: String, depth: usize) -> String {
    ciphercells_to_plaintext(&mut ciphertext_to_ciphercells(ciphertext, depth))
}

#[test]
fn test_encode() {
    assert_eq!(encode("helloworld".to_string(), 4), "hoewrlolld".to_string());
}

#[test]
fn test_decode() {
    assert_eq!(decode("hoewrlolld".to_string(), 4), "helloworld".to_string());
}

fn test_roundtrip() {
    // TODO: test roundtrip with randomly-generated strings
}
