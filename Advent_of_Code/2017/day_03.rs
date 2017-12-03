#![feature(conservative_impl_trait, inclusive_range_syntax)]

use std::iter;
use std::collections::HashMap;

fn spiral_instructions() -> impl Iterator<Item=char> {
    (1..).flat_map(|i| if i % 2 == 1 {
        iter::repeat('→').take(i).chain(iter::repeat('↑').take(i))
    } else {
        iter::repeat('←').take(i).chain(iter::repeat('↓').take(i))
    })
}

fn instructions_to_address<I: Iterator<Item=char>>(instructions: I) -> (isize, isize) {
    instructions.map(|i| match i {
        '↑' => (0, 1),
        '↓' => (0, -1),
        '←' => (-1, 0),
        '→' => (1, 0),
        _ => panic!("this is wrong"),
    }).fold((0, 0), |sum, delta| (sum.0 + delta.0, sum.1 + delta.1))
}

fn address_from_origin(pair: (isize,isize)) -> usize {
    let (x, y) = pair;
    (x.abs() + y.abs()) as usize
}

fn lookup(cell: usize) -> usize {
    address_from_origin(instructions_to_address(spiral_instructions().take(cell)))
}

fn neighborhood(x: isize, y: isize) -> Vec<(isize, isize)> {
    let mut folks = Vec::new();
    for i in -1..=1 {
        for j in -1..=1 {
            folks.push((x+i, y+j))
        }
    }
    folks
}

fn stress_test(threshold: usize) -> usize {
    let mut map = HashMap::new();
    let mut i = 1;
    loop {
        let (x, y) = instructions_to_address(spiral_instructions().take(i-1));
        let mut neighbor_total = 0;
        let mut hits = Vec::new();
        for neighbor in neighborhood(x, y) {
            if let Some(value) = map.get(&neighbor) {
                hits.push(neighbor.clone());
                neighbor_total += value;
            }
        }
        let inscription = if i == 1 { 1 } else { neighbor_total };
        println!("i={}, address={:?}, inscribing {} after hits {:?}",
                 i, (x, y), inscription, hits);
        map.insert((x, y), inscription);

        if neighbor_total > threshold {
            return neighbor_total
        }
        i += 1;
    }
}

fn main() {
    println!("{}", lookup(361527-1));
    println!("{}", stress_test(361527));
}
