import logging
import os

import torch

import utils.metric
from utils.helper import Mode
from setting import param_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RunModel:

    def __init__(self, name, net, dataset,
                 arch_lr, arch_lr_decay_milestones, arch_lr_decay_ratio, arch_decay, arch_clip_gradient,
                 weight_lr, weight_lr_decay_milestones, weight_lr_decay_ratio, weight_decay, weight_clip_gradient,
                 num_search_iterations, num_search_arch_samples,
                 num_train_iterations,
                 criterion, metric_names, metric_indexes,
                 print_frequency,
                 device_ids):

        self._name = name
        self._net = net.cuda()
        self._parallel_net = torch.nn.DataParallel(self._net, device_ids=device_ids)
        self._dataset = dataset

        # arch optimizer
        self._arch_lr = arch_lr
        self._arch_lr_decay_milestones = arch_lr_decay_milestones
        self._arch_lr_decay_ratio = arch_lr_decay_ratio
        self._arch_decay = arch_decay
        self._arch_clip_gradient = arch_clip_gradient

        # nn optimizer
        self._weight_lr = weight_lr
        self._weight_lr_decay_milestones = weight_lr_decay_milestones
        self._weight_lr_decay_ratio = weight_lr_decay_ratio
        self._weight_decay = weight_decay
        self._weight_clip_gradient = weight_clip_gradient

        self._num_search_iterations = num_search_iterations
        self._num_search_arch_samples = num_search_arch_samples
        self._num_train_iterations = num_train_iterations

        self._criterion = getattr(utils.metric, criterion)
        self._metric_names = metric_names
        self._metric_indexes = metric_indexes
        self._print_frequency = print_frequency
        self._device_ids = device_ids

    def load(self, mode):
        self.initialize()
        save_dir = os.path.join(param_path, self._name)
        filename = os.path.join(save_dir, '%s.pth' % mode)
        try:
            states = torch.load(filename)
            # load net
            self._net.load_state_dict(states['net'])
            # load optimizer
            self._arch_optimizer.load_state_dict(states['arch_optimizer'])
            self._arch_optimizer_scheduler.load_state_dict(states['arch_optimizer_scheduler'])
            self._weight_optimizer.load_state_dict(states['weight_optimizer'])
            self._weight_optimizer_scheduler.load_state_dict(states['weight_optimizer_scheduler'])
            # load historical records
            self._best_epoch = states['best_epoch']
            self._valid_records = states['valid_records']
            logging.info('load architecture [epoch %d] from %s [ok]', self._best_epoch, filename)
        except:
            logging.info('load architecture [fail]')
            logging.info('initialize the optimizer')
            self.initialize()

    def clear_records(self):
        self._best_epoch = -1
        self._valid_records = []

    def initialize(self):
        # initialize for weight optimizer
        self._weight_optimizer = torch.optim.Adam(
            self._net.weight_parameters(),
            lr=self._weight_lr,
            weight_decay=self._weight_decay
        )
        self._weight_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self._weight_optimizer,
            milestones=self._weight_lr_decay_milestones,
            gamma=self._weight_lr_decay_ratio,
        )
        # initialize for arch optimizer
        self._arch_optimizer = torch.optim.Adam(
            self._net.arch_parameters(),
            lr=self._arch_lr,
            weight_decay=self._arch_decay
        )
        self._arch_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self._arch_optimizer,
            milestones=self._arch_lr_decay_milestones,
            gamma=self._arch_lr_decay_ratio,
        )
        # initialize validation records
        self.clear_records()

    def _save(self, mode):
        save_dir = os.path.join(param_path, self._name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        states = {
            'net': self._net.state_dict(),
            'arch_optimizer': self._arch_optimizer.state_dict(),
            'arch_optimizer_scheduler': self._arch_optimizer_scheduler.state_dict(),
            'weight_optimizer': self._weight_optimizer.state_dict(),
            'weight_optimizer_scheduler': self._weight_optimizer_scheduler.state_dict(),
            'best_epoch': self._best_epoch,
            'valid_records': self._valid_records
        }
        filename = os.path.join(save_dir, '%s.pth' % mode)
        torch.save(obj=states, f=filename)
        logging.info('[eval]\tepoch[%d]\tsave parameters to %s', self._best_epoch, filename)

    def _save_checkpoint(self, epoch, mode):
        save_dir = os.path.join(param_path, self._name, 'checkpoint-%d' % epoch)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        states = {
            'net': self._net.state_dict(),
            'arch_optimizer': self._arch_optimizer.state_dict(),
            'arch_optimizer_scheduler': self._arch_optimizer_scheduler.state_dict(),
            'weight_optimizer': self._weight_optimizer.state_dict(),
            'weight_optimizer_scheduler': self._weight_optimizer_scheduler.state_dict(),
            'best_epoch': self._best_epoch,
            'valid_records': self._valid_records
        }
        filename = os.path.join(save_dir, '%s.pth' % mode)
        # torch.save(obj=states, f=filename)
        logging.info('save checkpoint-%d to %s', epoch, filename)

    def _add_record(self, metrics, mode):
        self._valid_records += [metrics.get_value()]
        if self._best_epoch < 0 or \
                self._valid_records[self._best_epoch][self._metric_names[0]] > self._valid_records[-1][
            self._metric_names[0]]:
            self._best_epoch = len(self._valid_records) - 1
            self._save(mode)

    def search_gradient_step(self, epoch):
        speedometer = Speedometer(
            title='search',
            epoch=epoch,
            metric_names=self._metric_names,
            metric_indexes=self._metric_indexes,
            print_frequency=self._print_frequency * len(self._device_ids),
            batch_size=self._dataset.batch_size_per_gpu
        )

        self._weight_optimizer.zero_grad()
        self._arch_optimizer.zero_grad()
        for i in range(self._num_search_iterations):

            self._net.train()
            for j in range(self._num_search_arch_samples // len(self._device_ids)):
                inputs, labels = next(iter(self._dataset.search_train))
                preds = self._parallel_net(inputs, Mode.TWO_PATHS)

                preds = self._dataset.scaler.inverse_transform(preds)
                loss = self._criterion(preds, labels)
                loss.backward(retain_graph=False)
                # log metrics
                speedometer.update(preds, labels, len(self._device_ids))

            torch.nn.utils.clip_grad_norm_(self._net.weight_parameters(), self._weight_clip_gradient)
            self._weight_optimizer.step()
            self._weight_optimizer.zero_grad()

            if epoch < 10: continue

            self._net.eval()
            for j in range(self._num_search_arch_samples // len(self._device_ids)):
                inputs, labels = next(iter(self._dataset.search_valid))
                preds = self._parallel_net(inputs, Mode.TWO_PATHS)
                preds = self._dataset.scaler.inverse_transform(preds)
                loss = self._criterion(preds, labels)
                loss.backward(retain_graph=False)

            torch.nn.utils.clip_grad_norm_(self._net.arch_parameters(), self._arch_clip_gradient)
            self._arch_optimizer.step()
            self._arch_optimizer.zero_grad()

        self._weight_optimizer_scheduler.step()
        
        if epoch >= 10:
            self._arch_optimizer_scheduler.step()

        return speedometer.finish()

    def train_gradient_step(self, epoch):
        speedometer = Speedometer(
            title='train',
            epoch=epoch,
            metric_names=self._metric_names,
            metric_indexes=self._metric_indexes,
            print_frequency=self._print_frequency,
            batch_size=self._dataset.batch_size_per_gpu
        )
        for i in range(self._num_train_iterations):
            # forward & backward
            self._net.train()
            self._weight_optimizer.zero_grad()
            inputs, labels = next(iter(self._dataset.train))
            preds = self._net(inputs.cuda(), Mode.ONE_PATH_FIXED)
            preds = self._dataset.scaler.inverse_transform(preds)
            loss = self._criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._net.weight_parameters(), self._weight_clip_gradient)
            self._weight_optimizer.step()
            # log metrics
            speedometer.update(preds, labels)

        self._weight_optimizer_scheduler.step()
        return speedometer.finish()

    def test(self, epoch, dataloader, tag):
        speedometer = Speedometer(
            title=tag,
            epoch=epoch,
            metric_names=self._metric_names,
            metric_indexes=self._metric_indexes,
            print_frequency=self._print_frequency,
            batch_size=self._dataset.batch_size_per_gpu
        )
        for nbatch, (inputs, labels) in enumerate(dataloader):
            with torch.no_grad():
                self._net.eval()
                preds = self._net(inputs.cuda(), Mode.ONE_PATH_FIXED)
                preds = self._dataset.scaler.inverse_transform(preds)
            # log metrics
            speedometer.update(preds, labels)

        return speedometer.finish()

    def search(self, num_epoch):
        print(self._net.__repr__())
        for epoch in range(self._best_epoch + 1, self._best_epoch + 1 + num_epoch):
            self.search_gradient_step(epoch)
            self._add_record(self.test(epoch, self._dataset.valid, tag='valid'), mode='search')
            self.test(epoch, self._dataset.test, tag='test')
            # if epoch % 10 == 0:
            #     self._save_checkpoint(epoch, mode='search')

    def train(self, num_epoch):
        for epoch in range(num_epoch):
            self.train_gradient_step(epoch)
            self._add_record(self.test(epoch, self._dataset.valid, tag='valid'), mode='train')
            self.test(epoch, self._dataset.test, tag='test')

        self.test(num_epoch - 1, self._dataset.valid, tag='valid')
        self.test(num_epoch - 1, self._dataset.test, tag='test')


import time


class Speedometer:
    def __init__(self, title, epoch, metric_names, metric_indexes, print_frequency, batch_size):
        self._title = title
        self._epoch = epoch
        self._metric_names = metric_names
        self._metric_indexes = metric_indexes
        self._print_frequency = print_frequency
        self._batch_size = batch_size
        self.reset()

    def reset(self):
        self._metrics = utils.metric.Metrics(self._metric_names, self._metric_indexes)
        self._start = time.time()
        self._tic = time.time()
        self._counter = 0

    def update(self, preds, labels, step_size=1):
        self._metrics.update(preds, labels)
        self._counter += step_size
        if self._counter % self._print_frequency == 0:
            time_spent = time.time() - self._tic
            speed = float(self._print_frequency * self._batch_size) / time_spent
            out_str = [
                '[%s]' % self._title,
                'epoch[%d]' % self._epoch,
                'batch[%d]' % self._counter,
                'time: %.2f' % time_spent,
                'speed: %.2f samples/s' % speed,
                str(self._metrics)
            ]
            logging.info('\t'.join(out_str))
            self._tic = time.time()

    def finish(self):
        out_str = [
            '[%s]' % self._title,
            'epoch[%d]' % self._epoch,
            'time: %.2f' % (time.time() - self._start),
            str(self._metrics)
        ]
        logging.info('\t'.join(out_str))
        return self._metrics
