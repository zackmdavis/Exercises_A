struct Instructions {
    cursor: isize,
    counter: usize,
    list: Vec<isize>,
}

impl Instructions {
    fn jump(&mut self) -> bool {
        let delta = self.list[self.cursor as usize];
        if delta >= 3 {
            self.list[self.cursor as usize] = self.list[self.cursor as usize] - 1;
        } else {
            self.list[self.cursor as usize] = self.list[self.cursor as usize] + 1;
        }
        self.cursor += delta;
        self.counter += 1;
        self.cursor < 0 || self.cursor >= (self.list.len() as isize)
    }
}

fn main() {
    let prompt = vec![0, 3, 0, 1, -3]; // actual puzzle input was really long
                                       // and was copy-pasted here
    let mut i = Instructions { cursor: 0, counter: 0, list: prompt.clone() };
    loop {
        let done = i.jump();
        if done {
            break;
        }
    }
    println!("{}", i.counter);
}
