# Function to generate the expected benefit of a classifier, as detailed here: 
# http://www.svds.com/the-basics-of-classifier-evaluation-part-1/
def ExpCost(y_true,y_pred,benefit_tP,cost_fN, benefit_tN, cost_fP):
    

    confArray = confusion_matrix(y_true, y_pred)
    tN = float(confArray[0][0]) # true negative (0)
    fP = float(confArray[0][1]) # false positive
    fN = float(confArray[1][0]) # false negative
    tP = float(confArray[1][1]) # true positive (1)

    prob_Pos = tP/(tP+tN) # prob of positive
    prob_Neg = 1.0 - prob_Pos # prob of negative

    # probabilities of the different terms
    prob_tP = tP/(tP + fP + fN + tN) 
    prob_fP = fP/(tP + fP + fN + tN)
    prob_fN = fN/(tP + fP + fN + tN)
    prob_tN = tN/(tP + fP + fN + tN)

    cost = prob_Pos*(prob_tP*benefit_tP + prob_fN*cost_fN) + prob_tN*(prob_tN*benefit_tN + prob_fP*cost_fP) # the expected cost
    return cost 