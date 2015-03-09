// "Describe how you could use a single array to implement three stacks."

// Uh, we could pretend the residue classes mod 3 of the array indices
// actually live in different arrays?

struct ThreeStackArray {
    space: [i32; 25],
    top1: i32,
    top2: i32,
    top3: i32
}

fn tsa_pop(tsa: ThreeStackArray, stack_index: usize) -> Option<i32> {
    // TODO
}
