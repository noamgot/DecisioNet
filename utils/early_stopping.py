from torch.optim.lr_scheduler import ReduceLROnPlateau

class EarlyStopping:
    """Early stops the training if a metric doesn't improve after a given patience."""

    def __init__(self,
                 mode: str = 'min',
                 patience: int = 15,
                 threshold: float = 1e-4,
                 threshold_mode: str = 'rel',
                 verbose: bool = False):
        """
        Args:
            mode: Either 'min' or 'max'.
                  Determines whether we want to minimize or maximize the desired metric. Default: 'min'.
            patience: Number of epochs to wait without an improvement before raising the early stop flag. Default: 15.
            threshold: Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
            threshold_mode: One of `rel`, `abs`. In `rel` mode, dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode. In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
            verbose: Whether to print informative messages. Default: False.
        """
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.patience = patience
        self.verbose = verbose
        self.num_bad_epochs = 0
        self.best = None
        self.early_stop = False
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold_mode=threshold_mode)
        self._reset()

    def _init_is_better(self, mode, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float("inf")
        else:  # mode == 'max':
            self.mode_worse = -float("inf")

    def _reset(self):
        """Resets num_bad_epochs counter."""
        self.best = self.mode_worse
        self.num_bad_epochs = 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def step(self, metrics):

        current = float(metrics)
        self.last_epoch += 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.num_bad_epochs} out of {self.patience}')
            if self.num_bad_epochs > self.patience:
                self.early_stop = True

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold_mode=self.threshold_mode)

    def state_dict(self):
        return self.__dict__


if __name__ == '__main__':
    es = EarlyStopping(verbose=True)
    for i in range(100):
        print(i)
        es.step(10)
        if es.early_stop:
            print("DONE!")
            break
