// What would it feel like to write a Tetris clone in C++? What if I were to
// sketch out part of this in my exercises repo (telling myself that I have the
// option of expanding it into an actual project later, even though I am
// unlikely to invest time in such a thing)?

// This is mostly born out of a feeling that no matter how good I get at Rust,
// even if Rust ends up winning the future, I'll always be inferior for having
// taking the Python track there rather than the grizzled C/C++ track

// The amount of time it takes me to do ludicrously simple things tells me that
// I don't already know C++ and that learning would be a substantial time
// investment, but we already knew that

#include <vector>
#include <iostream>

using namespace std;

class Piecemask {
private:
    vector<vector<bool> > mask;

public:
    Piecemask();
    Piecemask O();

    void Print() {
        for (int i=0; i < 5; i++) {
            for (int j=0; j < 5; j++) {
                if (mask[i][j]) {
                    cout << "X";
                } else {
                    cout << " ";
                }
            }
            cout << "\n";
        }
    }
};

// I J L O S T Z

Piecemask::Piecemask() {
    mask = vector<vector<bool> >(5, vector<bool>(5, false));
}

Piecemask Piecemask::O() {
    mask = vector<vector<bool> >(5, vector<bool>(5, false));
    for (int i=1; i < 2; i++) {
        for (int j=1; j < 2; j++) {
            mask[i][j] = true;
        }
    }
}

int main() {
    cout << "Hello world!\n";
    return 0;
}
