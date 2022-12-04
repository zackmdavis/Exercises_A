use std::fs;

fn parse_lines() -> Vec<String> {
    let content = fs::read_to_string("input").expect("file should exist");
    content.split("\n").map(|line| line.to_owned()).collect()
}

#[derive(Copy, Clone, Debug)]
struct Range {
    start: usize,
    end: usize,
}

fn parse_pair(line: String) -> (Range, Range) {
    let mut splitter = line.split(",");
    let raw_pair_1 = splitter.next().unwrap();
    let raw_pair_2 = splitter.next().unwrap();

    let mut pair_splitter_1 = raw_pair_1.split("-");
    let start_1: usize = pair_splitter_1.next().unwrap().parse().unwrap();
    let end_1: usize = pair_splitter_1.next().unwrap().parse().unwrap();

    let mut pair_splitter_2 = raw_pair_2.split("-");
    let start_2: usize = pair_splitter_2.next().unwrap().parse().unwrap();
    let end_2: usize = pair_splitter_2.next().unwrap().parse().unwrap();

    (Range{start: start_1, end: end_1}, Range{start: start_2, end: end_2})
}

fn the_first_star(lines: Vec<String>) -> u32 {
    let mut count = 0;
    for line in lines {
        let (pair_1, pair_2) = parse_pair(line);
        if (pair_1.start >= pair_2.start && pair_1.end <= pair_2.end) || (pair_2.start >= pair_1.start && pair_2.end <= pair_1.end) {
            count += 1;
        }
    }
    count
}

fn the_second_star(lines: Vec<String>) -> u32 {
    let mut count = 0;
    for line in lines {
        let (pair_1, pair_2) = parse_pair(line);
        if !(pair_1.end < pair_2.start || pair_2.end < pair_1.start) {
            count += 1;
        }
    }
    count
}

fn main() {
    let lines = parse_lines();
    println!("{}", the_first_star(lines.clone()));
    println!("{}", the_second_star(lines.clone()));
}
