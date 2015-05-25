// http://www.reddit.com/r/dailyprogrammer/comments/
// 2rnwzf/20150107_challenge_196_intermediate_rail_fence/

// Railfence cipher!

use std::cmp::Ordering;

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
        length += 1;
        if (depth == zig_depth - 1) || (depth == 0) {
            zig = !zig;
        }
        if zig {
            depth += 1
        } else {
            depth -= 1
        }
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

fn encode(plaintext: String) -> String {
    "TODO".to_string()
}

fn decode(ciphertext: String) -> String {
    "TODO".to_string()
}

#[test]
fn test_plaintext_to_ciphercells() {
    let plaintext: String = "bookaimstogetyouup".to_string();
    let expected_ciphercells: Vec<RailfenceCipherCell> = vec![
        RailfenceCipherCell{ figure: 'b', length: 0, depth: 0 },
        RailfenceCipherCell{ figure: 'o', length: 1, depth: 1 },
        RailfenceCipherCell{ figure: 'o', length: 2, depth: 2 },
        RailfenceCipherCell{ figure: 'k', length: 3, depth: 1 },
        RailfenceCipherCell{ figure: 'a', length: 4, depth: 0 },
        RailfenceCipherCell{ figure: 'i', length: 5, depth: 1 },
        RailfenceCipherCell{ figure: 'm', length: 6, depth: 2 },
        RailfenceCipherCell{ figure: 's', length: 7, depth: 1 },
        RailfenceCipherCell{ figure: 't', length: 8, depth: 0 },
        RailfenceCipherCell{ figure: 'o', length: 9, depth: 1 },
        RailfenceCipherCell{ figure: 'g', length: 10, depth: 2 },
        RailfenceCipherCell{ figure: 'e', length: 11, depth: 1 },
        RailfenceCipherCell{ figure: 't', length: 12, depth: 0 },
        RailfenceCipherCell{ figure: 'y', length: 13, depth: 1 },
        RailfenceCipherCell{ figure: 'o', length: 14, depth: 2 },
        RailfenceCipherCell{ figure: 'u', length: 15, depth: 1 },
        RailfenceCipherCell{ figure: 'u', length: 16, depth: 0 },
        RailfenceCipherCell{ figure: 'p', length: 17, depth: 1 },
    ];
    assert_eq!(expected_ciphercells, plaintext_to_ciphercells(plaintext, 3));
}

#[test]
fn test_ciphercells_to_ciphertext() {
    let mut ciphercells: Vec<RailfenceCipherCell> = vec![
        RailfenceCipherCell{ figure: 'r', length: 0, depth: 0 },
        RailfenceCipherCell{ figure: 'u', length: 1, depth: 1 },
        RailfenceCipherCell{ figure: 's', length: 2, depth: 2 },
        RailfenceCipherCell{ figure: 't', length: 3, depth: 1 },
        RailfenceCipherCell{ figure: 'y', length: 4, depth: 0 },
    ];
    let expected_ciphertext: String = "ryuts".to_string();
    assert_eq!(expected_ciphertext, ciphercells_to_ciphertext(&mut ciphercells));
}
