from sklearn import metrics
import numpy as np

def get_tpr_at_fpr(tpr_lst, fpr_lst, score_lst, fpr_value):

	abs_fpr = np.absolute(fpr_lst - fpr_value)
	idx = np.argmin(abs_fpr)
	while idx < len(abs_fpr) - 1 and abs_fpr[idx + 1] == abs_fpr[idx]:
		idx += 1
	return tpr_lst[idx], score_lst[idx]


def my_metrics(label_list, pred_list, val_phase=False):

	fpr, tpr, scores = metrics.roc_curve(label_list,pred_list,
										 drop_intermediate=True)
	auc_score = metrics.auc(fpr,tpr)
	fnr = 1 - tpr
	tnr = 1 - fpr
	EER0 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
	EER1 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
	EER  = min(EER0, EER1) 
	best_ACER, best_AP, best_BP = 100, 100, 100
	best_threshold = 100
	for idx_ in range(len(tpr)):		
		_tpr, _fpr = tpr[idx_], fpr[idx_]
		_tnr, _fnr = tnr[idx_], fnr[idx_]
		assert _tpr + _fnr == 1, print(_tpr, _fnr)
		assert _tnr + _fpr == 1, print(_tnr, _fpr)
		APCER = _fpr/(_fpr+_tnr)
		BPCER = _fnr/(_fnr+_tpr)
		ACER  = 0.5 * (APCER+BPCER)
		if ACER < best_ACER:
			best_ACER = ACER
			best_AP   = APCER
			best_BP   = BPCER
			best_threshold = scores[idx_]

	abs_fnr = np.absolute(fnr - 0.005)
	idx = np.argmin(abs_fnr)
	res_tpr = tpr[idx]
	if not val_phase:
		tpr_h, _ = get_tpr_at_fpr(tpr, fpr, scores, 0.005)	# 0.5%
		tpr_m, _ = get_tpr_at_fpr(tpr, fpr, scores, 0.05)	# 5%
		tpr_l, _ = get_tpr_at_fpr(tpr, fpr, scores, 0.01)	# 1%
		return best_AP, best_BP, best_ACER, EER, res_tpr, auc_score, [tpr_h, tpr_m, tpr_l]
	else:
		tpr_h, _ = get_tpr_at_fpr(tpr, fpr, scores, 0.005)	# 0.5%
		tpr_m, _ = get_tpr_at_fpr(tpr, fpr, scores, 0.05)	# 5%
		tpr_l, _ = get_tpr_at_fpr(tpr, fpr, scores, 0.01)	# 1%
		return best_AP, best_BP, best_ACER, EER, res_tpr, auc_score, [tpr_h, tpr_m, tpr_l], fpr, tpr
		