import numpy as np

def evaluate(H, Y, beta=1.0):
   tp = sum((Y == H) * (Y == 1) * 1)
   tn = sum((Y == H) * (Y == 0) * 1)
   fp = sum((Y != H) * (Y == 0) * 1)
   fn = sum((Y != H) * (Y == 1) * 1)

   accuracy = (tp + tn) / (tp + fp + fn + tn)
   sensitivity = tp / (tp + fn)
   specificity = tn / (fp + tn)
   precision = tp / (tp + fp)
   recall = sensitivity
   f_score = ( (beta**2 + 1) * precision * recall) / (beta**2 * precision + recall)
   auc = (sensitivity + specificity) / 2
   youden = sensitivity - (1 - specificity)
   #p_plus = sensitivity / (1 - specificity)
   p_minus = (1 - sensitivity) / specificity
   #dp = (np.sqrt(3) / np.pi) * (np.log(sensitivity/(1 - sensitivity) + np.log(specificity/(1 - specificity))))

   result = {}
   result["tp"] = tp
   result["tn"] = tn
   result["fp"] = fp
   result["fn"] = fn
   result["accuracy"] = accuracy
   result["sensitivity"] = sensitivity
   result["specificity"] = specificity
   result["precision"] = precision
   result["recall"] = recall
   result["f-score"] = f_score
   result["AUC"] = auc
   result["Youden"] = youden
   #result["p+"] = p_plus
   result["p-"] = p_minus
   #result["DP"] = dp

   return result