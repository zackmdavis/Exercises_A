use std::fs;

fn parse_lines() -> Vec<String> {
    let content = fs::read_to_string("input").expect("file should exist");
    content.split("\n").map(|line| line.to_owned()).collect()
}


fn the_first_star(lines: Vec<String>) -> u32 {
    0
}

fn the_second_star(lines: Vec<String>) -> u32 {
    0
}

fn main() {
    let lines = parse_lines();
    println!("{}", the_first_star(lines.clone()));
    println!("{}", the_second_star(lines.clone()));
}
