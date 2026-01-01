from .segmentation_tracker import SegmentationTracker
from sklearn.metrics import roc_curve, auc



class bhatkale45Tracker(SegmentationTracker):
    def __init__(self, dataset, stage: str = "train", wandb_log: bool = False, use_tensorboard: bool = False, dataset_opt: dict=None):
        """ Segmentation tracker bhatkale45 problem. The dataset needs to have a
        class_to_segment member that defines how metrics get computed and agregated.
        It follows shapenet official formula for computing miou which treats missing part as having an iou of 1
        See https://github.com/charlesq34/pointnet2/blob/42926632a3c33461aebfbee2d829098b30a23aaa/part_seg/evaluate.py#L166-L176

        Arguments:
            dataset {[type]}

        Keyword Arguments:
            stage {str} -- current stage (default: {"train"})
            wandb_log {bool} -- Log to Wanndb (default: {False})
            use_tensorboard {bool} -- Log to tensorboard (default: {False})
        """
        super(bhatkale45Tracker, self).__init__(dataset=dataset, stage=stage, wandb_log=wandb_log, use_tensorboard=use_tensorboard, dataset_opt=dataset_opt)

#    def _compute_metrics(self, outputs, labels):
#        super()._compute_metrics(outputs, labels)
#        self._fpr = dict()
#        self._tpr = dict()
#        self._roc_auc = dict()
#        for i in range(self._num_classes):
#            self._fpr[i], self._tpr[i], _ = roc_curve(labels.cpu(), outputs[:, i].cpu(), pos_label=i)
#            self._roc_auc[i] = auc(self._fpr[i], self._tpr[i])
#

    
