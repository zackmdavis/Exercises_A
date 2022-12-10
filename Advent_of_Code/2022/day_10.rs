use std::fs;

fn parse_lines() -> Vec<String> {
    let content = fs::read_to_string("input").expect("file should exist");
    content.split("\n").map(|line| line.to_owned()).collect()
}

#[derive(Copy, Clone, Debug)]
enum Operation {
    NoOp,
    AddX(i32),
}

fn lines_to_instructions(lines: Vec<String>) -> Vec<Operation> {
    let mut operations = vec![];
    for line in lines {
        if line == "" {
            break;
        }
        if line == "noop" {
            operations.push(Operation::NoOp);
        } else {
            let mut splitter = line.split(" ");
            splitter.next(); // `addx `
            let n: i32 = splitter.next().unwrap().parse().unwrap();
            operations.push(Operation::AddX(n))
        }
    }
    operations
}

fn the_first_star(operations: Vec<Operation>) -> i32 {
    let mut clock = 0;
    let mut x = 1;

    fn check_signal(clock: u32, x: i32) -> Option<i32> {
        // 20th, 60th, 100th, 140th, 180th, and 220th cycles
        match clock {
            20 => Some(20 * x),
            60 => Some(60 * x),
            100 => Some(100 * x),
            140 => Some(140 * x),
            180 => Some(180 * x),
            220 => Some(220 * x),
            _ => None
        }
    }

    let mut strengths = vec![];

    for operation in operations {
        match operation {
            Operation::NoOp => {
                clock += 1;
                if let Some(strength) = check_signal(clock, x) {
                    strengths.push(strength)
                }
            },
            Operation::AddX(n) => {
                clock += 1;
                if let Some(strength) = check_signal(clock, x) {
                    strengths.push(strength)
                }
                clock += 1;
                if let Some(strength) = check_signal(clock, x) {
                    strengths.push(strength)
                }
                x += n;
            }
        }
    }
    strengths.iter().sum()
}

fn the_second_star(operations: Vec<Operation>) -> String {
    let mut output = String::new();
    let mut line = String::new();
    let mut clock = 0;
    let mut x = 1;

    fn check_pixel(clock: i32, x: i32) -> bool {
        let position = (clock - 1) % 40;
        (position - 1 <= x) && (x <= position + 1)
    }

    for operation in operations {
        match operation {
            Operation::NoOp => {
                clock += 1;
                if check_pixel(clock, x) {
                    line.push('#');
                } else {
                    line.push('.');
                }
                if line.len() == 40 {
                    output.push('\n');
                    output.push_str(&line);
                    line = String::new();
                }
            },
            Operation::AddX(n) => {
                clock += 1;
                if check_pixel(clock, x) {
                    line.push('#');
                } else {
                    line.push('.');
                }
                if line.len() == 40 {
                    output.push('\n');
                    output.push_str(&line);
                    line = String::new();
                }

                clock += 1;
                if check_pixel(clock, x) {
                    line.push('#');
                } else {
                    line.push('.');
                }
                if line.len() == 40 {
                    output.push('\n');
                    output.push_str(&line);
                    line = String::new();
                }
                x += n;
            }
        }
    }
    output
}

fn main() {
    let lines = parse_lines();
    let operations = lines_to_instructions(lines);
    println!("{}", the_first_star(operations.clone()));
    println!("{}", the_second_star(operations.clone()));
}
