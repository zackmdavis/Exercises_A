use std::fs;

fn parse_elf_provisions() -> Vec<Vec<u32>> {
    let content = fs::read_to_string("input").expect("file should exist");
    let lines = content.split("\n");
    let mut packs = vec![];
    let mut current_pack = vec![];
    for line in lines {
        if line == "" {
            packs.push(current_pack);
            current_pack = vec![];
        } else {
            current_pack.push(line.parse().expect("lines should be integer"));
        }
    }
    packs
}

fn the_first_star(packs: Vec<Vec<u32>>) -> u32 {
    let mut max = 0;
    for pack in packs {
        let mut calories = 0;
        for item in pack {
            calories += item;
        }
        if calories > max {
            max = calories;
        }
    }
    max
}

use std::collections::BinaryHeap;

fn the_second_star(packs: Vec<Vec<u32>>) -> u32 {
    let mut maxes = BinaryHeap::new();
    for pack in packs {
        let mut calories = 0;
        for item in pack {
            calories += item;
        }
        maxes.push(calories);
    }
    let mut total = 0;
    for _ in 0..3 {
        total += maxes.pop().expect("heap item should exist");
    }
    total
}

fn main() {
    let packs = parse_elf_provisions();
    println!("{}", the_first_star(packs.clone()));
    println!("{}", the_second_star(packs.clone()));
}
