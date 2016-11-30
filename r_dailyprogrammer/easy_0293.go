package main

import "log"

// https://www.reddit.com/r/dailyprogrammer/comments/5e4mde/20161121_challenge_293_easy_defusing_the_bomb/

// """
// To disarm the bomb you have to cut some wires. These wires are either white, black, purple, red, green or orange.

// The rules for disarming are simple:

// If you cut a white cable you can't cut white or black cable.
// If you cut a red cable you have to cut a green one
// If you cut a black cable it is not allowed to cut a white, green or orange one
// If you cut a orange cable you should cut a red or black one
// If you cut a green one you have to cut a orange or white one
// If you cut a purple cable you can't cut a purple, green, orange or white cable

// If you have anything wrong in the wrong order, the bomb will explode.

// [...]

// A state machine will help this make easy
// """

// gr9 hint

type cable int

const (
	nullary cable = iota // this programming language sucks (need option types)
	white
	black
	purple
	red
	green
	orange
)

func isOkayToCut(thisCable cable, previousCable cable) bool {
	switch previousCable {
	case white:
		return thisCable != white && thisCable != black
	case black:
		return thisCable != white && thisCable != green && thisCable != orange
	case purple:
		return thisCable != white && thisCable != green && thisCable != orange && thisCable != purple
	case red:
		return thisCable == green
	case green:
		return thisCable == orange || thisCable == white
	case orange:
		return thisCable == red || thisCable == black
	default:
		return true
	}
}

func isSafeCutSequence(theseCables []cable) bool {
	previousCable := nullary
	for _, thisCable := range theseCables {
		if !isOkayToCut(thisCable, previousCable) {
			return false
		}
		previousCable = thisCable
	}
	return true
}

func main() {
	firstInput := []cable{white, red, green, white}
	secondInput := []cable{white, orange, green, white}

	if !isSafeCutSequence(firstInput) {
		log.Fatalf("expected first input sequence to be safe")
	}

	if isSafeCutSequence(secondInput) {
		log.Fatalf("expected second input sequence to be unsafe")
	}

	log.Println("success, probably")
}
