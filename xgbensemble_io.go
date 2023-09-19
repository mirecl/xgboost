package xgboost

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
)

type xgboostJSONModel struct {
	NodeID                int                 `json:"nodeid,omitempty"`
	SplitFeatureID        string              `json:"split,omitempty"`
	SplitFeatureThreshold float64             `json:"split_condition,omitempty"`
	YesID                 int                 `json:"yes,omitempty"`
	NoID                  int                 `json:"no,omitempty"`
	MissingID             int                 `json:"missing,omitempty"`
	LeafValue             float64             `json:"leaf,omitempty"`
	Children              []*xgboostJSONModel `json:"children,omitempty"`
}

func loadFeatureMap(featureFile io.Reader) (map[string]int, error) {
	var featureMap map[string]int

	b, err := io.ReadAll(featureFile)
	if err != nil {
		return map[string]int{}, err
	}

	err = json.Unmarshal(b, &featureMap)
	if err != nil {
		return map[string]int{}, err
	}

	return featureMap, nil
}

func convertFeatToIdx(featureMap map[string]int, feature string) (int, error) {
	if _, ok := featureMap[feature]; !ok {
		return 0, fmt.Errorf("cannot find feature %s in feature map", feature)
	}
	return featureMap[feature], nil
}

func buildTree(xgbTreeJSON *xgboostJSONModel, maxDepth int, featureMap map[string]int) (*xgbTree, int, error) {
	stack := make([]*xgboostJSONModel, 0)
	maxFeatIdx := 0
	t := &xgbTree{}
	stack = append(stack, xgbTreeJSON)
	var node *xgbNode
	var maxNumNodes int
	var maxIdx int
	if maxDepth != 0 {
		maxNumNodes = int(math.Pow(2, float64(maxDepth+1)) - 1)
		t.nodes = make([]*xgbNode, maxNumNodes)
	}
	for len(stack) > 0 {
		stackData := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if stackData.Children == nil {
			// leaf node.
			node = &xgbNode{
				NodeID:     stackData.NodeID,
				Flags:      isLeaf,
				LeafValues: stackData.LeafValue,
			}
		} else {
			featIdx, err := convertFeatToIdx(featureMap, stackData.SplitFeatureID)
			if err != nil {
				return nil, 0, err
			}
			if featIdx > maxFeatIdx {
				maxFeatIdx = featIdx
			}
			node = &xgbNode{
				NodeID:    stackData.NodeID,
				Threshold: stackData.SplitFeatureThreshold,
				No:        stackData.NoID,
				Yes:       stackData.YesID,
				Missing:   stackData.MissingID,
				Feature:   featIdx,
			}
			// find real length of the tree.
			if maxDepth != 0 {
				t := int(math.Max(float64(stackData.NoID), float64(stackData.YesID)))
				if t > maxIdx {
					maxIdx = t
				}
			}
			stack = append(stack, stackData.Children...)
		}
		if maxNumNodes > 0 {
			if node.NodeID >= maxNumNodes {
				return nil, 0, fmt.Errorf("wrong tree max depth %d, please check your model again for the"+
					" correct parameter", maxDepth)
			}
			t.nodes[node.NodeID] = node
		} else {
			// do not know the depth beforehand just append.
			t.nodes = append(t.nodes, node)
		}
	}
	if maxDepth == 0 {
		sort.SliceStable(t.nodes, func(i, j int) bool {
			return t.nodes[i].NodeID < t.nodes[j].NodeID
		})
	} else {
		t.nodes = t.nodes[:maxIdx+1]
	}

	return t, maxFeatIdx, nil
}

// XGEnsembleFromFile loads xgboost model from json file.
func XGEnsembleFromFile(modelPath, featuresPath string) (Ensembler, error) {
	modelReader, err := os.Open(modelPath)
	if err != nil {
		return nil, err
	}
	defer modelReader.Close()

	featuresReader, err := os.Open(featuresPath)
	if err != nil {
		return nil, err
	}
	defer featuresReader.Close()

	return XGEnsembleFromReader(modelReader, featuresReader)
}

// XGEnsembleFromReader loads xgboost model from reader.
func XGEnsembleFromReader(modelReader, featuresReader io.Reader) (Ensembler, error) {
	var xgbEnsembleJSONModel []*xgboostJSONModel

	dec := json.NewDecoder(modelReader)
	err := dec.Decode(&xgbEnsembleJSONModel)
	if err != nil {
		return nil, err
	}

	featMap, err := loadFeatureMap(featuresReader)
	if err != nil {
		return nil, err
	}

	nTrees := len(xgbEnsembleJSONModel)
	if nTrees == 0 {
		return nil, fmt.Errorf("no trees in file")
	}

	e := &xgbEnsemble{}
	e.Trees = make([]*xgbTree, 0, nTrees)
	maxFeat := 0

	for i := 0; i < nTrees; i++ {
		tree, numFeat, err := buildTree(xgbEnsembleJSONModel[i], 4, featMap)
		if err != nil {
			return nil, fmt.Errorf("error while reading %d tree: %s", i, err.Error())
		}
		e.Trees = append(e.Trees, tree)
		if numFeat > maxFeat {
			maxFeat = numFeat
		}
	}
	e.numFeat = maxFeat + 1
	return e, nil
}
