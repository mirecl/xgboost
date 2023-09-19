package xgboost

import (
	"errors"
)

// Ensembler contains interface of a base model.
type Ensembler interface {
	Predict(features []float64) (float64, error)
}

type xgbEnsemble struct {
	Trees      []*xgbTree
	numClasses int
	numFeat    int
}

// Predict returns prediction of this ensemble model.
func (e *xgbEnsemble) Predict(features []float64) (float64, error) {
	if e.numClasses == 0 {
		return 0, errors.New("0 class please check your model")
	}
	if e.numClasses != 1 {
		return 0, errors.New("regression prediction only support binary classes for now")
	}
	// number of trees for 1 class.
	pred := 0.0
	numTreesPerClass := len(e.Trees) - 1
	for k := 0; k < numTreesPerClass; k++ {
		p, err := e.Trees[k*e.numClasses].predict(features)
		if err != nil {
			return 0, err
		}
		pred += p
	}

	return pred + 0.5, nil
}
