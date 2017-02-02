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

#[derive(Debug, PartialEq)]
enum Cell {
    Pacgum(usize),
    Wall,
    Warp,
}

#[derive(Clone, Debug, PartialEq)]
struct Pacmap {
    cursor: (isize, isize),
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
        Pacmap { cursor: (0, 0), rows: rows, cols: cols, cells: vec![] }
    }

    fn construct(spec: &str) -> Result<Self, PacmapConstructionError> {
        let mut cells = vec![];
        let mut row_counter = 0;
        let mut col_width_slot = None;
        let mut cursor_slot = None;
        for (i, line) in spec.split('\n').enumerate() {
            row_counter += 1;
            for (j, cellspec) in line.chars().enumerate() {
                let cell = match cellspec {
                    'C' => {
                        cursor_slot = Some((i, j));
                        Cell::Pacgum(0)
                    },
                    n @ '0'...'9' => {
                        let gumno = n.to_string().parse().expect("should not match if not digit");
                        Cell::Pacgum(gumno)
                    },
                    'X' => Cell::Wall,
                    'O' => Cell::Warp,
                    s @ _ => {
                        return Err(PacmapConstructionError::new(&format!("invalid cell spec: {:?}", s)))
                    }
                };
                cells.push(cell);
            }
            // set column width if it hasn't been set (i.e., we just processed
            // the first row)
            if col_width_slot.is_none() {
                col_width_slot = Some(cells.len());
            }
            // check column width
            if cells.len() != col_width_slot.expect("col_width_slot should be set") * row_counter {
                return Err(PacmapConstructionError::new(&format!("dimensional mismatch")))
            }
        }

        Ok(Pacmap {
            cursor: cursor_slot.expect("cusor_slot should be set"),
            rows: row_counter,
            cols: col_width_slot.expect("col_width_slot should be set"),
            cells: cells
        })
    }

    fn look(&mut self) -> Cell {
        &mut self.cells[rows * self.cursor.0 + self.cursor.1]
    }

    fn act(&self, action: (isize, isize)) -> Option<Self> {
        let mut advance = self.clone();
        advance.cursor = (advance.cursor.0 + action.0, advance.cursor.1 + action.1);
        if !(advance.cursor.0 >= 0 &&
             advance.cursor.0 < advance.rows &&
             advance.cursor.1 >= 0 &&
             advance.cursor.1 < advance.cols) {
            return None
        }
        match advance.look() {
            Cell::Pacgum(mut ref n) => {
                if n > 0 { // eat gum
                    n -= 1
                }
            },
            Cell::Wall => { return None; },
            Cell::Warp => {
                // ummm, need to think about how to handle warps; prompt note
                // says "you can either choose to ignore it or teleport
                // yourself"
                //
                // program is not even compiling right now
            }
        }
    }
}

fn pacsearch(pacmap: Pacmap, ticks: usize) -> usize {
    // do some sort of recursive backtracking search
    for action in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)].iter() {
        // TODO ...
    }

    0
}

fn main() {
    // TODO
}

#[test]
fn concerning_example_input_one_map_construction() {
    let example_input = "\
XXXXX
X197X
X2C6X
X345X
XXXXX";
    let constructed = Pacmap::construct(example_input).expect("should be able to construct Pacmap");
    let expected = Pacmap {
        cursor: (2, 2),
        rows: 5,
        cols: 5,
        cells: vec![Cell::Wall, Cell::Wall,      Cell::Wall,      Cell::Wall,      Cell::Wall,
                    Cell::Wall, Cell::Pacgum(1), Cell::Pacgum(9), Cell::Pacgum(7), Cell::Wall,
                    Cell::Wall, Cell::Pacgum(2), Cell::Pacgum(0), Cell::Pacgum(6), Cell::Wall,
                    Cell::Wall, Cell::Pacgum(3), Cell::Pacgum(4), Cell::Pacgum(5), Cell::Wall,
                    Cell::Wall, Cell::Wall,      Cell::Wall,      Cell::Wall,      Cell::Wall]
    };
    assert_eq!(expected, constructed);
}

#[test]
fn concerning_erroneous_map_spec_construction_error() {
    let spec = "\
XXXXX
XC97X
X26ZX
";
    let attempt = Pacmap::construct(spec);
    assert_eq!(Err(PacmapConstructionError::new("invalid cell spec: 'Z'")),
               attempt);
}
