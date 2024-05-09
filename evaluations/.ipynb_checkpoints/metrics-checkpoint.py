import numpy as np

class Metrics():
    """
    calculate evaluate metrics of surgical workflow recognization: ACC, Recall, Precision, Jaccard
    
    ACC: mean and std acc of each video
    Recall: first calculate mean recall of each video, then calculate mean and std recall of each phase
    Precision: first calculate mean precision of each video, then calculate mean and std precision of each phase
    Jaccard: first calculate mean jaccard of each video, then calculate mean and std jaccard of each phase
    """

    def __init__(self, num_classes=7):
        self.jaccard_list, self.precision_list, self.recall_list, self.acc_list = [], [], [], []
        self.num_classes = num_classes

    def evaluate(self, gtLabelID, predLabelID):
        jacc_list, recall_list, prec_list = [np.nan]*self.num_classes, [np.nan]*self.num_classes, [np.nan]*self.num_classes

        for iPhase in range(self.num_classes):
            pred_idx_list = np.argwhere(predLabelID==iPhase).reshape(-1) 
            label_idx_list = np.argwhere(gtLabelID==iPhase).reshape(-1) 
            
            if len(label_idx_list) == 0:
                continue
            
            iPUnion = len(np.union1d(pred_idx_list, label_idx_list))
            tp = len(np.intersect1d(pred_idx_list, label_idx_list))
            
            jaccard = tp / iPUnion
            jaccard = jaccard * 100
            jacc_list[iPhase] = jaccard

            sumPred = np.sum(predLabelID == iPhase)
            prec_list[iPhase] = tp * 100 / sumPred
            
            sumGT = np.sum(gtLabelID == iPhase)
            recall_list[iPhase]  = tp * 100 / sumGT

        acc = np.sum(gtLabelID==predLabelID) / len(gtLabelID)
        acc = acc * 100
        
        return jacc_list, prec_list, recall_list, acc

    def add_video_sample(self, video_id, gtLabelID, predLabelID):
        """ gtLabelID, predLabelID: np.array, [1, seq_len] """
        res, prec, rec, acc = self.evaluate(gtLabelID, predLabelID) 
        self.jaccard_list.append(np.array(res))
        self.precision_list.append(np.array(prec))
        self.recall_list.append(np.array(rec))
        self.acc_list.append(acc)
        return
    
    def jaccard(self):
        jaccard = np.array(self.jaccard_list)
        jaccard[jaccard > 100] = 100
        meanJaccPerPhase = np.nanmean(jaccard, 0)
        meanJacc = np.nanmean(meanJaccPerPhase)
        stdJacc = np.std(meanJaccPerPhase)
        meanjaccphase, stdjaccphase = [], []
        for h in range(self.num_classes):
            jaccphase = jaccard[:, h]
            meanjaccphase.append(np.nanmean(jaccphase))
            stdjaccphase.append(np.nanstd(jaccphase))
        return meanJacc, stdJacc, meanjaccphase, stdjaccphase

    def precision(self):
        precision = np.array(self.precision_list)
        precision[precision > 100] = 100
        meanPrecPerPhase = np.nanmean(precision, 0)
        meanPrec = np.nanmean(meanPrecPerPhase)
        stdPrec = np.nanstd(meanPrecPerPhase)
        meanprecphase, stdprecphase = [], []
        for h in range(self.num_classes):
            precphase = precision[:, h]
            meanprecphase.append(np.nanmean(precphase))
            stdprecphase.append(np.nanstd(precphase))
        return meanPrec, stdPrec, meanprecphase, stdprecphase

    def recall(self):
        recall = np.array(self.recall_list)
        recall[recall > 100] = 100
        meanRecPerPhase = np.nanmean(recall, 0)
        meanRec = np.nanmean(meanRecPerPhase)
        stdRec = np.nanstd(meanRecPerPhase)
        meanrecphase, stdrecphase = [], []
        for h in range(self.num_classes):
            recphase = recall[:, h]
            meanrecphase.append(np.nanmean(recphase))
            stdrecphase.append(np.nanstd(recphase))
        return meanRec, stdRec, meanrecphase, stdrecphase

    def acc(self):
        acc_list = np.array(self.acc_list)
        meanAcc = np.mean(acc_list)
        stdAcc = np.std(acc_list)
        return meanAcc, stdAcc

    def clear(self):
        self.jaccard_list, self.precision_list, self.recall_list, self.acc_list = [], [], [], []
        return
    
    def metrics_dict(self, prefix=""):
        return {
            prefix + 'acc': self.acc()[0],
            prefix + 'recall': self.recall()[0],
            prefix + 'precision': self.precision()[0],
            prefix + 'jaccard': self.jaccard()[0],
        }










