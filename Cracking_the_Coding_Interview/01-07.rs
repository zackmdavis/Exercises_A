// "Write an algorithm such that if an element in an MxN matrix is 0,
// its entire row and column are set to 0.


fn zero_row(matrix: Vec<int>, row: uint) {
    // XXX: cannot index a value of type `collections::vec::Vec<int>`
    let row_length = matrix[0].len();
    for j in range(0i, row_length) {
        matrix[row][j] = 0;
    }
}

fn zero_col(matrix: Vec<int>, col: uint) {
    let col_length = matrix.len();
    for i in range(0i, col_length) {
        matrix[i][col] = 0;
    }
}

fn determinator(matrix: Vec<int>) {

}

fn zero_row_demonstration() {
    // let matrix = vec![vec![1, 2],
    //                   vec![3, 4],];
    let mut matrix = Vec::new();
    matrix.push(vec![1, 2]);
    matrix.push(vec![3, 4]);

    zero_row(matrix, 0);
    println!("{}", matrix);
}

fn main() {
    zero_row_demonstration();
}
