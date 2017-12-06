#![feature(inclusive_range_syntax)]

use std::collections::HashMap;

struct Memory {
    banks: [usize; 16],
    previous_states: HashMap<[usize; 16], usize>
}

impl Memory {
    fn reallocate(&mut self) -> Option<usize> {
        let fullest_bank_index = {
            let max = self.banks.iter().max().unwrap();
            self.banks.iter().enumerate().find(|&(_, b)| b == max).unwrap().0
        };
        let blocks_to_redistribute = self.banks[fullest_bank_index];
        self.banks[fullest_bank_index] = 0;
        for block in 1..=blocks_to_redistribute {
            self.banks[(fullest_bank_index+block) % 16] += 1;
        }
        let state_index = self.previous_states.len();
        self.previous_states.insert(self.banks.clone(), state_index)
    }

    fn new(banks: &[usize; 16]) -> Self {
        let mut previous_states = HashMap::new();
        previous_states.insert(banks.clone(), 0);
        Memory { banks: banks.clone(), previous_states }
    }
}

fn reallocation_collapse_time(mut m: Memory) -> usize {
    let mut reallocations = 0;
    while let None = m.reallocate() {
        reallocations += 1;
    }
    reallocations + 1
}

fn reallocation_cycle_len(mut m: Memory) -> usize {
    loop {
        match m.reallocate() {
            None => { continue; },
            Some(first_seen_index) => {
                let now_index = m.previous_states.get(&m.banks).unwrap();
                return now_index - first_seen_index;
            }
        }
    }
}

fn main() {
    println!("{}",
             reallocation_collapse_time(
                 Memory::new(&[11, 11, 13, 7, 0, 15, 5, 5,
                               4, 4, 1, 1, 7, 1, 15, 11])));

    println!("{}",
             reallocation_cycle_len(
                 Memory::new(&[11, 11, 13, 7, 0, 15, 5, 5,
                               4, 4, 1, 1, 7, 1, 15, 11])));

}
