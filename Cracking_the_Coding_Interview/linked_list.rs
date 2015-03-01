// You are not expected to compile this

pub struct LinkedList<T> {
    head: Option<LinkedListNode<T>>,
}

pub struct LinkedListNode<T> {
    next: Option<Box<LinkedListNode<T>>>,
    datum: T,
}

trait LinkedListNature {
    fn append_to_tail<T>(&self, value: T) -> ();
}

impl<T> LinkedListNature for LinkedList<T> {
    fn append_to_tail<U>(&self, value: U) -> () {
        let mut at_the_end = false;
        let mut inspecting = &self.head;
        while !at_the_end {
            inspecting = match inspecting {
                Some(n) => n.next,
                None => { at_the_end = true; None }
            }
        }
    }
}

fn main() {}
