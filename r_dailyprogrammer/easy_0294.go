package main

import "log"

// https://www.reddit.com/r/dailyprogrammer/comments/5go843/20161205_challenge_294_easy_rack_management_1/

// Today's challenge is inspired by the board game Scrabble. Given a set of 7 letter tiles and a word, determine whether you can make the given word using the given tiles.

// Optional Bonus 1: Handle blank tiles (represented by "?"). These are "wild card" tiles that can stand in for any single letter.

type rack map[rune]uint

func buildRack(rackview string) rack {
	ourRack := make(rack)
	for _, r := range rackview {
		ourRack[r]++
	}
	return ourRack
}

func spellable(rackview string, target string) bool {
	ourRack := buildRack(rackview)
	for _, r := range target {
		supply, ok := ourRack[r]
		if !ok || supply == 0 {
			blankSupply, blankOk := ourRack['?']
			if !blankOk || blankSupply == 0 {
				return false
			} else {
				ourRack['?']--
			}
		} else {
			ourRack[r]--
		}
	}
	return true
}

type expectation struct {
	rackview  string
	target    string
	spellable bool
}

func main() {
	expectations := []expectation{
		expectation{rackview: "ladilmy", target: "daily", spellable: true},
		expectation{rackview: "eerriin", target: "eerie", spellable: false},
		expectation{rackview: "orrpgma", target: "program", spellable: true},
		expectation{rackview: "orppgma", target: "program", spellable: false},
		expectation{rackview: "piizza?", target: "pizzazz", spellable: false},
		expectation{rackview: "pizza??", target: "pizzazz", spellable: true},
		expectation{rackview: "a??????", target: "program", spellable: true},
		expectation{rackview: "b??????", target: "program", spellable: false},
	}
	for _, e := range expectations {
		if spellable(e.rackview, e.target) != e.spellable {
			var negatoryMaybe string
			if !e.spellable {
				negatoryMaybe = "not "
			}
			log.Fatalf("expected %vto be able to spell %q with %q", negatoryMaybe, e.target, e.rackview)
		}
	}
	log.Println("WINNING")
}
