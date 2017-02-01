// https://www.reddit.com/r/dailyprogrammer/comments/5iq4ix/20161216_challenge_295_hard_advanced_pacman/

// This challenge takes its roots from the world-famous game Pacman. To finish
// the game, pacman needs to gather all pacgum on the map.

// The goal of this chalenge is to have a time-limited pacman. Pacman must
// gather as much pacgum as possible in the given time. To simplify, we will
// say that 1 move (no diagonals) = 1 unit of time.

// Input description

// You will be given a number, the time pacman has to gather as much pacgum as
// possible, and a table, being the map pacman has to explore. Every square of
// this map can be one of those things :

// A number N between (1 and 9) of pacgums that pacman can gather in one unit
// of time.

// "X" squares cannot be gone through.

// "C" will be where pacman starts.

// "O" (the letter, not zero ) will be a warp to another "O". There can be only
// 2 "O" on one map; Output description

// Your program should output the maximum number of pacgums pacman can gather
// in the given time.

use std::error::Error;
use std::fmt::{self, Display};

#[derive(Debug)]
enum Cell {
    Pacgum(usize),
    Wall,
    Warp,
}

#[derive(Debug)]
struct Pacmap {
    rows: usize,
    cols: usize,
    cells: Vec<Cell>,
}

#[derive(Debug)]
struct PacmapConstructionError {
    description: String
}

impl PacmapConstructionError {
    fn new(description: &str) -> Self {
        PacmapConstructionError { description: description.to_owned() }
    }
}


impl Display for PacmapConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Error: {}", self.description())
    }
}

impl Error for PacmapConstructionError {
    fn description(&self) -> &str {
        &self.description
    }
}

impl Pacmap {
    fn new(rows: usize, cols: usize) -> Self {
        Pacmap { rows: rows, cols: cols, cells: vec![] }
    }

    fn construct(spec: &str) -> Result<Self, PacmapConstructionError> {
        let mut cells = vec![];
        let mut row_counter = 0;
        let mut col_width = None;
        for line in spec.split('\n') {
            row_counter += 1;
            for cellspec in line.chars() {
                let cell = match cellspec {
                    // TODO: 'C' for heroine start location
                    n @ '0'...'9' => {
                        let gumno = n.to_string().parse().expect("should not match if not digit");
                        Cell::Pacgum(gumno)
                    },
                    'X' => Cell::Wall,
                    'O' => Cell::Warp,
                    s @ _ => {
                        return Err(PacmapConstructionError::new(&format!("invalid cell spec {}", s)))
                    }
                };
                cells.push(cell);
            }
            // set column width if it hasn't been set (i.e., we just processed
            // the first row)
            if col_width.is_none() {
                col_width = Some(cells.len());
            }
            // check column width
            if cells.len() != col_width.expect("col_width should be set") * row_counter {
                return Err(PacmapConstructionError::new(&format!("dimensional mismatch")))
            }
        }

        Ok(Pacmap {
            rows: row_counter,
            cols: col_width.expect("col_width should be set"),
            cells: cells
        })
    }
}


fn main() {
    // TODO
}

#[test]
fn test_example_input_one() {
    // TODO: restore "C" for start location
    let example_input = "\
XXXXX
X197X
X206X
X345X
XXXXX";
    let _map = Pacmap::construct(example_input).expect("should be able to construct Pacmap");
}
