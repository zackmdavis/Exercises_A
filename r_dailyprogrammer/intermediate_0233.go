// https://www.reddit.com/r/dailyprogrammer/comments/3m2vvk/20150923_challenge_233_intermediate_game_of_text/

// Conway Life, except that live cells hold a rune, and a newly-alive
// cell randomly (uniform distribution) inherits a rune from one of its
// three neighbor-parents

package main;

import (
	"fmt"
	"math/rand"
	"time"
)

var the_fates = rand.New(rand.NewSource(time.Now().UnixNano()))

func darwinize(creature rune, community_survey []rune) rune {
	survivors := make([]rune, 8)
	be_counted := 0
	for _, cell := range community_survey {
		if cell != 0 {
			survivors[be_counted] = cell
			be_counted += 1
		}
	}
	if creature != 0 {
		if be_counted == 2 || be_counted == 3 {
			return creature;
		} else {
			return 0
		}
	} else {
		if be_counted == 3 {
			return survivors[int(the_fates.Float64() * float64(be_counted))]
		} else {
			return 0
		}
	}
}


func administer_survey(universe [][]rune, south int, east int) (results []rune) {
	southbreadth := len(universe)
	eastbreadth := len(universe[0])

	results = make([]rune, 8)
	placed := 0
	for i := -1; i <= 1; i++ {
		for j:= -1; j <= 1; j++ {
			if i == 0 && j == 0 {
				continue
			}
			if south + i >= 0 && east + j >= 0 && south + i < southbreadth &&
				east + j < eastbreadth {
				results[placed] = universe[south+i][east+j]
				placed++
			}
		}
	}
	return
}


func advance(universe [][]rune) (new_universe [][]rune) {
	southbreadth := len(universe)
	eastbreadth := len(universe[0])
	new_universe = make([][]rune, southbreadth)
	for i := 0; i < southbreadth; i++ {
		new_universe[i] = make([]rune, eastbreadth)
	}

	for i, row := range universe {
		for j, cell := range row {
			survey := administer_survey(universe, i, j)
			legacy := darwinize(cell, survey)
			new_universe[i][j] = legacy
		}
	}
	return
}


func chart(universe [][]rune) {
	fmt.Println("——————————————————")
	for _, row := range universe {
		for _, atom_cup := range row {
			if atom_cup == 0 {
				fmt.Printf(" ")
			} else {
				fmt.Printf("%c", atom_cup)
			}
		}
		fmt.Println("")
	}
	fmt.Println("——————————————————")
}


func main() {
	universe := make([][]rune, 20)
	for i := 0; i < 20; i++ {
		universe[i] = make([]rune, 20)
	}

	// 01234
	//0
	//1   o
	//2    o
	//3  ooo
	universe[1][3] = 'a'
	universe[2][4] = 'b'
	universe[3][2] = 'c'
	universe[3][3] = 'd'
	universe[3][4] = 'e'

	for {
		chart(universe)
		universe = advance(universe)
		time.Sleep(time.Second)
	}
	return
}
