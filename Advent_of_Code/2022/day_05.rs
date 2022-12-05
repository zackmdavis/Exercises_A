use std::fs;

// going to manually edit the input to avoid some annoying parsing

fn parse_lines() -> Vec<String> {
    let content = fs::read_to_string("input").expect("file should exist");
    content.split("\n").map(|line| line.to_owned()).collect()
}

fn starting_stacks() -> Vec<Vec<char>> {
    vec![
        vec!['D', 'H', 'N', 'Q', 'T', 'W', 'V', 'B'],
        vec!['D', 'W', 'B'],
        vec!['T', 'S', 'Q', 'W', 'J', 'C'],
        vec!['F', 'J', 'R', 'N', 'Z', 'T', 'P'],
        vec!['G', 'P', 'V', 'J', 'M', 'S', 'T'],
        vec!['B', 'W', 'F', 'T', 'N'],
        vec!['B', 'L', 'D', 'Q', 'F', 'H', 'V', 'N'],
        vec!['H', 'P', 'F', 'R'],
        vec!['Z', 'S', 'M', 'B', 'L', 'N', 'P', 'H']
    ]
}


fn the_first_star(lines: Vec<String>) -> Vec<char> {
    let mut stacks = starting_stacks();
    for line in lines {
        let mut words = line.split_whitespace();
        assert_eq!(Some("move"), words.next());
        let n: usize = words.next().unwrap().parse().unwrap();
        assert_eq!(Some("from"), words.next());
        let src: usize = words.next().unwrap().parse().unwrap();
        assert_eq!(Some("to"), words.next());
        let dst: usize = words.next().unwrap().parse().unwrap();
        for i in 0..n {
            let cargo = stacks[src-1].pop().unwrap();
            stacks[dst-1].push(cargo);
        }
    }
    let mut message = vec![];
    for mut stack in stacks {
        message.push(stack.pop().unwrap());
    }
    message
}

use std::collections::VecDeque;

fn the_second_star(lines: Vec<String>) -> Vec<char> {
    let mut stacks = starting_stacks();
    for line in lines {
        let mut words = line.split_whitespace();
        assert_eq!(Some("move"), words.next());
        let n: usize = words.next().unwrap().parse().unwrap();
        assert_eq!(Some("from"), words.next());
        let src: usize = words.next().unwrap().parse().unwrap();
        assert_eq!(Some("to"), words.next());
        let dst: usize = words.next().unwrap().parse().unwrap();

        let mut substack = VecDeque::new();
        for _ in 0..n {
            let cargo = stacks[src-1].pop().unwrap();
            substack.push_front(cargo);
        }
        for _ in 0..n {
            let cargo = substack.pop_front().unwrap();
            stacks[dst-1].push(cargo);
        }
    }

    let mut message = vec![];
    for mut stack in stacks {
        message.push(stack.pop().unwrap());
    }
    message
}

fn main() {
    let lines = parse_lines();
    println!("{:?}", the_first_star(lines.clone()));
    println!("{:?}", the_second_star(lines.clone()));
}
