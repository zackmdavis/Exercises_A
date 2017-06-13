struct Machine {
    running: bool,
    instruction_pointer: usize,
    instruction_counter: usize,
    registers: [usize; 10],
    memory: [usize; 1000]
}

impl Machine {
    fn new() -> Self {
        Machine { running: false,
                  instruction_pointer: 0,
                  instruction_counter: 0,
                  registers: [0; 10],
                  memory: [0; 1000] }
    }

    fn do_instruction(&mut self, instruction: usize) {
        println!("doing instruction {}", instruction);
        let opcode = instruction / 100;
        let arg1 = (instruction / 10) % 10;
        let arg2 = instruction % 10;

        self.instruction_pointer += 1;
        self.instruction_counter += 1;

        match opcode {
            0 => {
                self.jump(arg1, arg2);
            },
            1 => {
                self.running = false;
            },
            2 => {
                self.set_register(arg1, arg2);
            },
            3 => {
                self.add_to_register(arg1, arg2);
            },
            4 => {
                self.multiply_register(arg1, arg2);
            },
            5 => {
                self.set_register_to_register(arg1, arg2);
            },
            6 => {
                self.add_to_register_from_register(arg1, arg2);
            },
            7 => {
                self.multiply_register_by_register(arg1, arg2);
            },
            8 => {
                self.pull_from_ram(arg1, arg2);
            },
            9 => {
                self.push_to_ram(arg1, arg2);
            },
            _ => panic!("invalid opcode")
        }
    }

    fn load_program(&mut self, program: &[usize]) {
        for (i, &instruction) in program.iter().enumerate() {
            self.memory[i] = instruction;
        }
    }

    fn run(&mut self) {
        self.running = true;
        while self.running {
            let instruction = self.memory[self.instruction_pointer];
            self.do_instruction(instruction);
        }
    }

    // 2dn
    fn set_register(&mut self, d: usize, n: usize) {
        self.registers[d] = n;
    }

    // 3dn
    fn add_to_register(&mut self, d: usize, n: usize) {
        self.registers[d] += n;
    }

    // 4dn
    fn multiply_register(&mut self, d: usize, n: usize) {
        self.registers[d] *= n;
    }

    // 5ds
    fn set_register_to_register(&mut self, d: usize, s: usize) {
        self.registers[d] = self.registers[s];
    }

    // 6ds
    fn add_to_register_from_register(&mut self, d: usize, s: usize) {
        self.registers[d] += self.registers[s];
    }

    // 7ds
    fn multiply_register_by_register(&mut self, d: usize, s: usize) {
        self.registers[d] *= self.registers[s];
    }

    // 8da
    fn pull_from_ram(&mut self, d: usize, a: usize) {
        self.registers[d] = self.memory[self.registers[a]];
    }

    // 9sa
    fn push_to_ram(&mut self, s: usize, a: usize) {
        self.memory[self.registers[a]] = self.registers[s];
    }

    // 0ds
    fn jump(&mut self, d: usize, s: usize) {
        if self.registers[s] != 0 {
            println!("setting instruction pointer to {}, the content of \
                      register no. {}", self.registers[d], d);
            self.instruction_pointer = self.registers[d];
        }
    }
}

fn main() {
    let mut our_machine = Machine::new();
    our_machine.load_program(&[299, 492, 495, 399, 492, 495, 399, 283,
                               279, 689, 078, 100, 000, 000, 000]);
    our_machine.run();
    // XXX we get an infinite loop, but the text said it was only supposed to
    // run 16 instructions
    //
    // it's not like this is even a hard exercise; how do you get this wrong,
    // dummy
    assert_eq!(16, our_machine.instruction_counter);
}
