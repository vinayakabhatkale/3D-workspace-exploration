import torch
import hydra
import numpy as np
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer
from torch_points3d.datasets.segmentation.bhatkale45 import bhatkale45
from torch_geometric.data.dataloader  import DataLoader
from torch_points3d.datasets.segmentation.utils.viz_pcl import Visualizations
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import os
from time import perf_counter_ns

import pandas as pd

import yaml
from pathlib import Path

#import debugpy
#debugpy.listen(5678) # listen on port 5678, choose the port how you like
#debugpy.wait_for_client() # wait for vs code to attach to process

# evaluation
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

def plot_roc_for_all_classes(tpr: dict, fpr: dict, n_classes: int, roc_auc: dict):
    lw = 2
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.show()

def plot_roc_single_class(fpr: dict, tpr: dict, roc_auc: dict, class_: int):
    plt.figure()
    lw = 2
    plt.plot(
        fpr[class_],
        tpr[class_],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[class_],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()


def roc(y: np.ndarray, y_pred: np.ndarray, n_classes: int = 4):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_binary = label_binarize(y, classes=np.arange(n_classes))
    y_pred_binary = label_binarize(y_pred, classes=np.arange(n_classes))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_binary[:, i], y_pred_binary[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i]) 
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_binary.ravel(), y_pred_binary.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plot_roc_for_all_classes(tpr=tpr, fpr=fpr, n_classes=4, roc_auc=roc_auc)
    
@hydra.main(config_path="conf", config_name="eval")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))

    best_model = torch.load(f'{os.path.join(cfg.checkpoint_dir, cfg.model_name)}.pt')
    trainer = Trainer(cfg)
    model = trainer._model
    model.eval()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number parameters {params}')
    
    # not only latest but also best_acc, best_macc and best_miou is possible
    #model.load_state_dict(best_model['models']['latest'])
    dataset_opt= yaml.safe_load(Path('/home/developer/bhatkale45/install/bhatkale45_environment_detection/share/bhatkale45_environment_detection/bhatkale45-deepviewaggregation/conf/data/segmentation/bhatkale45.yaml').read_text())
    camera = dataset_opt['camera']
    if dataset_opt['include_real_data_to_training'] is True:
        path = f'/home/developer/bhatkale45/install/bhatkale45_environment_detection/share/bhatkale45_environment_detection/bhatkale45-deepviewaggregation/torch_points3d/datasets/segmentation/utils/data/bhatkale45/real_in_train/zivid/torch_training_data_test.torchds'
    else:
        pass
    
    path = '/home/developer/bhatkale45/install/bhatkale45_environment_detection/share/bhatkale45_environment_detection/bhatkale45-deepviewaggregation/torch_points3d/datasets/segmentation/utils/data/bhatkale45/no_real_in_train/zivid/torch_training_data_test.torchds'

    ds_list = torch.load(path)

    data_list = list()
    #viz = Visualizations()
    for data_point, label in ds_list:
        xyz = data_point[:, :3]
        features = data_point[:, 3:6]
        data = Data(x=features, pos=xyz, coords=xyz, y=label)
        data_list.append(data)

    device = 'cuda'
    test_dataloader = DataLoader(data_list, batch_size=2)

    y = list()
    y_pred = list()

    list_times = list()
    for data in test_dataloader:
    #for data in trainer._dataset.test_dataloaders[0]:
        model.set_input(data, device=device)

        start = perf_counter_ns()
        model()
        end = perf_counter_ns()
        list_times.append(end-start)
        outputs = model.output
        
        pred_labels = np.argmax(outputs.cpu().detach().numpy(), 1)
        viz = Visualizations()
        points = data['coords'].cpu().detach().numpy()
        ground_truth = data['y'].cpu().detach().numpy()
        #y.append(np.expand_dims(ground_truth, axis=0))
        y.append(ground_truth)
        #y_pred.append(np.expand_dims(pred_labels, axis=0))
        y_pred.append(pred_labels)
        cm  = confusion_matrix(ground_truth, pred_labels, labels=np.arange(4))
        accuracy = cm.diagonal()/cm.sum(axis=1)
        print(f'Accuracy {accuracy}')
        disp = ConfusionMatrixDisplay(cm)
        #disp.plot()
        #plt.show()

        #viz.visualize_point_cloud(points=points, point_labels=pred_labels, show_labels=True)

    print(f'Mean inference time for model: {np.mean(list_times) * pow(10, -9)} +/- {np.std(list_times) * pow(10, -9)}')

    y_np = np.concatenate(y, axis=0)
    y_pred_np = np.concatenate(y_pred, axis=0)
    cm  = confusion_matrix(y_np, y_pred_np, labels=np.arange(4))
    accuracy = cm.diagonal()/cm.sum(axis=1)
    accuracy = [round(acc*100, 2) for acc in accuracy]
    df = pd.DataFrame.from_dict({camera: accuracy}, orient='index')
    df.to_csv('/home/andi/DeepViewAgg/camera_comparison.csv', mode='a', header=False)
    
    print(f'Overall Accuracy {accuracy}')

    #roc(y=y_np, y_pred=y_pred_np)

    # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0

def viz_original_point_cloud(path: str='training_data0.npy'):
    points = np.load(path)

    pcl_to_viz = 95
    viz = Visualizations()
    viz.visualize_point_cloud(points=points[pcl_to_viz][:,1:4], point_labels=points[pcl_to_viz][:,0], colors=points[pcl_to_viz][:,4:], show_labels=True)

if __name__ == "__main__":
    main()

