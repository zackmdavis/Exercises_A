use std::fs;

fn parse_lines() -> Vec<String> {
    let content = fs::read_to_string("input").expect("file should exist");
    content.split("\n").map(|line| line.to_owned()).collect()
}

#[derive(Copy, Clone)]
enum Play {
    Rock,
    Paper,
    Scissors,
}

impl Play {
    fn score(&self) -> u32 {
        match self {
            Play::Rock => 1,
            Play::Paper => 2,
            Play::Scissors => 3,
        }
    }
}

#[derive(Copy, Clone)]
struct Round {
    them: Play,
    me: Play
}

#[derive(Copy, Clone)]
enum Outcome {
    Win,
    Loss,
    Tie,
}

impl Round {
    fn score(&self) -> u32 {
        self.me.score() + match self.outcome() {
            Outcome::Win => 6,
            Outcome::Tie => 3,
            Outcome::Loss => 0,
        }
    }

    fn outcome(&self) -> Outcome {
        match (self.them, self.me) {
            (Play::Rock, Play::Paper) => Outcome::Win,
            (Play::Rock, Play::Scissors) => Outcome::Loss,
            (Play::Rock, Play::Rock) => Outcome::Tie,
            (Play::Paper, Play::Rock) => Outcome::Loss,
            (Play::Paper, Play::Paper) => Outcome::Tie,
            (Play::Paper, Play::Scissors) => Outcome::Win,
            (Play::Scissors, Play::Rock) => Outcome::Win,
            (Play::Scissors, Play::Paper) => Outcome::Loss,
            (Play::Scissors, Play::Scissors) => Outcome::Tie,
        }
    }
}

fn parse_line_for_the_first_star(line: String) -> Round {
    let mut marks = line.split(" ");
    let them = match marks.next() {
        Some("A") => Play::Rock,
        Some("B") => Play::Paper,
        Some("C") => Play::Scissors,
        a => panic!("shouldn't happen {:?}", a)
    };
    let me = match marks.next() {
        Some("X") => Play::Rock,
        Some("Y") => Play::Paper,
        Some("Z") => Play::Scissors,
        a => panic!("shouldn't happen {:?}", a)
    };
    Round { them, me }
}


fn the_first_star(lines: Vec<String>) -> u32 {
    let mut rounds = vec![];
    for line in lines {
        let round = parse_line_for_the_first_star(line);
        rounds.push(round)
    }
    let mut total = 0;
    for round in rounds {
        total += round.score();
    }
    total
}


fn parse_line_for_the_second_star(line: String) -> Round {
    let mut marks = line.split(" ");
    let them = match marks.next() {
        Some("A") => Play::Rock,
        Some("B") => Play::Paper,
        Some("C") => Play::Scissors,
        a => panic!("shouldn't happen {:?}", a)
    };
    let desired_outcome = match marks.next() {
        Some("X") => Outcome::Loss,
        Some("Y") => Outcome::Tie,
        Some("Z") => Outcome::Win,
        a => panic!("shouldn't happen {:?}", a)
    };
    let me = match (them, desired_outcome) {
        (Play::Rock, Outcome::Win) => Play::Paper,
        (Play::Rock, Outcome::Loss) => Play::Scissors,
        (Play::Rock, Outcome::Tie) => Play::Rock,
        (Play::Paper, Outcome::Win) => Play::Scissors,
        (Play::Paper, Outcome::Loss) => Play::Rock,
        (Play::Paper, Outcome::Tie) => Play::Paper,
        (Play::Scissors, Outcome::Win) => Play::Rock,
        (Play::Scissors, Outcome::Loss) => Play::Paper,
        (Play::Scissors, Outcome::Tie) => Play::Scissors,
    };
    Round { them, me }
}


fn the_second_star(lines: Vec<String>) -> u32 {
    let mut rounds = vec![];
    for line in lines {
        let round = parse_line_for_the_second_star(line);
        rounds.push(round)
    }
    let mut total = 0;
    for round in rounds {
        total += round.score();
    }
    total
}

fn main() {
    let lines = parse_lines();
    println!("{}", the_first_star(lines.clone()));
    println!("{}", the_second_star(lines.clone()));
}
