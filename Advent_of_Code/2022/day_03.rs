use std::fs;

fn parse_lines() -> Vec<String> {
    let content = fs::read_to_string("input").expect("file should exist");
    content.split("\n").map(|line| line.to_owned()).collect()
}


fn priority(letter: char) -> u32 {
    // https://stackoverflow.com/a/45344045
    let lowercase = ('a'..='z').collect::<Vec<_>>();
    let uppercase = ('A'..='Z').collect::<Vec<_>>();
    let mut alphabet = vec![];
    alphabet.extend(lowercase);
    alphabet.extend(uppercase);
    for (i, c) in alphabet.iter().enumerate() {
        if *c == letter {
            return (i + 1) as u32;
        }
    }
    panic!("should have returned in loop")
}

fn parse_rucksack(line: String) -> char {
    let length = line.len();
    let divider = length / 2;
    for c in line[..divider].chars() {
        for d in line[divider..].chars() {
            if c == d {
                return c
            }
        }
    }
    panic!("should have returned in loop")
}


fn the_first_star(lines: Vec<String>) -> u32 {
    let mut total = 0;
    for line in lines {
        let c = parse_rucksack(line);
        total += priority(c);
    }
    total
}

use std::collections::HashSet;

fn the_second_star(lines: Vec<String>) -> u32 {
    let mut total = 0;
    for i in 0..lines.len()/3 {
        let sack1 = &lines[3*i];
        let sack2 = &lines[3*i+1];
        let sack3 = &lines[3*i+2];

        let mut set1 = HashSet::new();
        for c in sack1.chars() {
            set1.insert(c);
        }

        let mut set2 = HashSet::new();
        for c in sack2.chars() {
            set2.insert(c);
        }

        let mut set3 = HashSet::new();
        for c in sack3.chars() {
            set3.insert(c);
        }

        // https://www.reddit.com/r/rust/comments/5v35l6/intersection_of_more_than_two_sets/
        let common = set1.iter().filter(|c| set2.contains(c)).filter(|c| set3.contains(c)).next().expect("intersection should exist");
        total += priority(*common);
    }
    total
}

fn main() {
    let lines = parse_lines();
    println!("{}", the_first_star(lines.clone()));
    println!("{}", the_second_star(lines.clone()));
}
