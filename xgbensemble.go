package xgboost

// Ensembler contains interface of a base model.
type Ensembler interface {
	Predict(features []float64) (float64, error)
}

type xgbEnsemble struct {
	Trees   []*xgbTree
	numFeat int
}

// Predict returns prediction of this ensemble model.
func (e *xgbEnsemble) Predict(features []float64) (float64, error) {
	// number of trees for 1 class.
	pred := 0.0
	numTreesPerClass := len(e.Trees) - 1
	for k := 0; k < numTreesPerClass; k++ {
		p, err := e.Trees[k].predict(features)
		if err != nil {
			return 0, err
		}
		pred += p
	}
	return pred + 0.5, nil
}
