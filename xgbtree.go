package xgboost

import (
	"errors"
	"fmt"
	"math"
)

// xgbtree constant values.
const (
	isLeaf = 1
)

var (
	ErrNilNode      = errors.New("nil node")
	ErrFeatureCount = errors.New("features less than `node.Feature`")
)

type xgbNode struct {
	NodeID     int
	Threshold  float64
	Yes        int
	No         int
	Missing    int
	Feature    int
	Flags      uint8
	LeafValues float64
}

type xgbTree struct {
	nodes []*xgbNode
}

func (t *xgbTree) predict(features []float64) (float64, error) {
	idx := 0
	for {
		node := t.nodes[idx]
		if node == nil {
			return 0, ErrNilNode
		}
		if node.Flags&isLeaf > 0 {
			return node.LeafValues, nil
		}

		if len(features) < node.Feature {
			return 0, fmt.Errorf("%w, count features %d, count `node.Feature` %d", ErrFeatureCount, len(features), node.Feature)
		}

		v := features[node.Feature]
		if math.IsNaN(v) {
			// missing value will be represented as NaN value.
			idx = node.Missing
		} else if v >= node.Threshold {
			idx = node.No
		} else {
			idx = node.Yes
		}
	}
}
