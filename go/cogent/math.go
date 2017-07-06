package cogent

import "math"

type rand struct {
	state []uint32
}

func (r *rand) next() uint32 {
	t := r.state[3]
	t ^= t << 11
	t ^= t >> 8
	r.state[3] = r.state[2]
	r.state[2] = r.state[1]
	r.state[1] = r.state[0]

	t ^= r.state[0]
	t ^= r.state[0] >> 19
	r.state[0] = t
	return t
}

func (r *rand) float64() float64 {
	return float64(r.next()) / float64(math.MaxUint32)
}

func (r *rand) nextMax(max int) int {
	return int(r.next() % uint32(max))
}

func (r *rand) nextRange(min, max int) int {
	diff := max - min
	return r.nextMax(diff) + min
}

var r = rand{
	state: []uint32{7919, 104729, 1299709, 15485863},
}
