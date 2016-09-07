import matplotlib.pyplot as plt
import numpy as np

pred_path = "pred.npy"
real_path = "real.npy"


pred = np.load(pred_path)
real = np.load(real_path)
print real.shape
print pred.shape
fpr = list()
tpr = list()

for th in range(49999900, 500000000, 1):
	thres = th / 10000000.0
	
	tp = 0
	fp = 0
	fn = 0
	tn = 0
	cnt = 0
	pos_cnt = 0
	for i in range(pred.shape[0]):
		if pred[i] > thres:
			pos_cnt += 1
			if pos_cnt > 184967:
				continue
			if real[i] == 1:
				tp += 1
			else:
				fp += 1
		else:
			if real[i] == 1:
				fn += 1
			else:
				tn += 1
	print fp,tn
 	print thres,fp*1.00/(fp+tn)
	fpr.append(fp*1.00/(fp+tn))
	tpr.append(tp*1.00/(tp+fn))

p1 = plt.subplot(111)
p1.plot(fpr, tpr)
p1.axis([0.0,1.0,0.0,1.0])
p1.set_ylabel("TPR")
p1.set_xlabel("FPR")
p1.set_title("Roc Curve")
plt.show()
anc = np.trapz(tpr, fpr)
print anc
