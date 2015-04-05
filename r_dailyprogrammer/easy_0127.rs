// http://www.reddit.com
// /r/dailyprogrammer/comments
// /1f7qp5/052813_challenge_127_easy_mccarthy_91_function/

// XXX HOW DO YOU DO (how do you say?) GENERICS IN THIS WORTHLESS
// EXCUSE FOR A LANGUAGE
//
// IN MY HOMELAND, WE DON'T EVEN HAVE A WORD FOR "GENERICS" BECAUSE
// EVERYTHING IS MADE OF DUCKS LIKE JHWH CONWAY INTENDED
//
// fn mccarthys_ninety_first<N: Int>(n: T) -> T {
fn mccarthys_ninety_first(n: isize) -> isize {
    if n > 100 {
        println!("{} because {} is greater than 100", n - 10, n);
        n - 10
    } else {
        // XXX? maybe the problem-setter intended us to print out what
        // the recursive expression would be instead of just doing it, as
        // here? How easy should an easy problem be?
        println!("M(M({})) because {} is not greater than 100", n, n);
        mccarthys_ninety_first(mccarthys_ninety_first(n + 11))
    }
}

fn main() {
    mccarthys_ninety_first(80);
}

#[test]
fn test_mccarthys_ninety_first() {
    for i in -5..102 {
        assert_eq!(91, mccarthys_ninety_first(i))
    }
    for i in 1..10 {
        assert_eq!(91 + i, mccarthys_ninety_first(101 + i))
    }
}
