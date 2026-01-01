import os
import numpy as np
from collections import defaultdict
import torchnet as tnt
import torch
from typing import Dict, Any
import wandb
from torch.utils.tensorboard import SummaryWriter
import logging
import pandas as pd

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.models import model_interface

log = logging.getLogger(__name__)


def meter_value(meter, dim=0):
    return float(meter.value()[dim]) if meter.n > 0 else 0.0


class BaseTracker:
    def __init__(self, stage: str, wandb_log: bool, use_tensorboard: bool, dataset_opt: dict=None ):
        self._wandb = wandb_log
        self._use_tensorboard = use_tensorboard
        self._tensorboard_dir = os.path.join(os.getcwd(), "tensorboard")
        self._n_iter = 0
        self._finalised = False
        self._conv_type = None
        self._history = list()
        self.dataset_opt = dataset_opt

        if self._use_tensorboard:
            log.info(
                "Access tensorboard with the following command <tensorboard --logdir={}>".format(self._tensorboard_dir)
            )
            self._writer = SummaryWriter(log_dir=self._tensorboard_dir)

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, value):
        self._history.extend(value)

    def reset(self, stage="train"):
        self._stage = stage
        self._loss_meters = {}
        self._finalised = False

    def store_metrics_for_kfold(self):
        self._history.append(self.get_metrics(verbose=True))

    def get_metrics(self, verbose=False, final: bool=False) -> Dict[str, Any]:
        metrics = {}
        for key, loss_meter in self._loss_meters.items():
            value = meter_value(loss_meter, dim=0)
            if value:
                metrics[key] = meter_value(loss_meter, dim=0)
        return metrics

    def finalize_cross_val_metrics(self):
        metrics = dict()
        for fold in self._history:
            for key, value in fold.items():
                if key not in metrics:
                    metrics[key] = list()
                metrics[key].append(value)

        self.cross_val_final_metrics = dict()
        
        for metric, value in metrics.items():
            if 'per_class' in metric:
                result_per_class = dict()
                for class_ in value[0].keys():
                    results = [float(d[class_]) for d in value]
                    best_model = results.index(max(results))
                    result_per_class[class_] = (np.mean(results), best_model)
                self.cross_val_final_metrics[metric] = result_per_class
            else:
                best_model_fn = min if 'loss' in metric else max
                best_model = value.index(best_model_fn(value))
                self.cross_val_final_metrics[metric] = (np.mean(value), best_model)

        if self._wandb:
            self.log_cross_validation_results()

    @property
    def metric_func(self):
        self._metric_func = {"loss": min}
        return self._metric_func

    def log_cross_validation_results(self):
        publish_dict = dict()
        for metric, value in self.cross_val_final_metrics.items():
            if 'per_class' in metric:
                continue

            publish_dict[f"cross_{metric}"] = value[0]
        publish_dict["best_model"] = self.get_best_model_from_cross_val_metrics()
        wandb.log(publish_dict)

    def track(self, model: model_interface.TrackerInterface, **kwargs):
        if self._finalised:
            raise RuntimeError("Cannot track new values with a finalised tracker, you need to reset it first")
        losses = self._convert(model.get_current_losses())
        self._append_losses(losses)

    def finalise(self, *args, **kwargs):
        """ Lifcycle method that is called at the end of an epoch. Use this to compute
        end of epoch metrics.
        """
        self._finalised = True

    def _append_losses(self, losses):
        for key, loss in losses.items():
            if loss is None:
                continue
            loss_key = "%s_%s" % (self._stage, key)
            if loss_key not in self._loss_meters:
                self._loss_meters[loss_key] = tnt.meter.AverageValueMeter()
            self._loss_meters[loss_key].add(loss)

    @staticmethod
    def _convert(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        else:
            return x

    def publish_to_tensorboard(self, metrics, step):
        for metric_name, metric_value in metrics.items():
            metric_name = "{}/{}".format(metric_name.replace(self._stage + "_", ""), self._stage)
            self._writer.add_scalar(metric_name, metric_value, step)

    @staticmethod
    def _remove_stage_from_metric_keys(stage, metrics):
        new_metrics = {}
        for metric_name, metric_value in metrics.items():
            new_metrics[metric_name.replace(stage + "_", "")] = metric_value
        return new_metrics

    def publish(self, step):
        """ Publishes the current metrics to wandb and tensorboard
        Arguments:
            step: current epoch
        """
        metrics = self.get_metrics()

        if self._wandb:
            wandb.log(metrics, step=step)

        if self._use_tensorboard:
            self.publish_to_tensorboard(metrics, step)

        # Some metrics may be intended for wandb or tensorboard
        # tracking but not for final final model selection. Those are
        # the metrics absent from self.metric_func and must be excluded
        # from the output of self.publish
        current_metrics = {
            k: v
            for k, v in self._remove_stage_from_metric_keys(self._stage, metrics).items()
            if k in self.metric_func.keys()}

        return {
            "stage": self._stage,
            "epoch": step,
            "current_metrics": current_metrics,
        }

    def get_best_model_from_cross_val_metrics(self):
        metrics = self.get_metrics(final=True)
        best_model_counter = defaultdict(int)
        for key, value in metrics.items():
            if 'per_class' in key:
                for results in value.items():
                    best_model_counter[results[1][1]] = best_model_counter[results[1][1]] + 1
            else:
                best_model_counter[value[1]] = best_model_counter[value[1]] + 1
        best_model = max(best_model_counter, key=best_model_counter.get)
        return best_model

    def print_summary(self, final: bool=False):
        metrics = self.get_metrics(verbose=True, final=final)
        if final:
            log.info("")
            log.info("")

        log.info("".join(["=" for i in range(50)]))
        if final:
            log.info("          Final Results of cross validation        ")
            log.info("".join(["=" for i in range(50)]))
            for key, value in metrics.items():
                if 'per_class' in key:
                    log.info(f"Results for {key}")
                    for class_, results in value.items():
                        log.info(" class {} = {}, best model: {}".format(class_, results[0], results[1]))

                    log.info("")
                else:
                    try:
                        log.info("    {} = {}, best model: {}".format(key, value[0], value[1]))
                    except:
                        # the only thing that fails here is when value is the timestamp
                        # we don't care about that
                        pass
            log.info("Best model in general: {}".format(self.get_best_model_from_cross_val_metrics())) 
            log.info("")
        else:
            metrics['real_data'] = self.dataset_opt['include_real_data_to_training']
            metrics['num_points'] = self.dataset_opt['number_points']
            metrics['model'] = self.dataset_opt['model']
            df = pd.DataFrame(metrics)
            df.to_csv('/home/developer/deepviewaggregation/torch_points3d/evaluation.csv', mode='a', index=False, header=False)
            for key, value in metrics.items():
                log.info("    {} = {}".format(key, value))

        log.info("".join(["=" for i in range(50)]))

    @staticmethod
    def _dict_to_str(dictionnary):
        string = "{"
        for key, value in dictionnary.items():
            string += "%s: %.2f," % (str(key), value)
        string += "}"
        return string
