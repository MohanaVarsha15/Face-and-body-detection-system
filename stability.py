
from collections import deque


class ExponentialSmoother:
    """Smooth fluctuating numerical values without freezing real changes."""

    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if new_value is None:
            return None

        new_value = float(new_value)

        if self.value is None:
            self.value = new_value
        else:
            self.value = (
                self.alpha * new_value
                + (1 - self.alpha) * self.value
            )

        return round(self.value, 3)


class StableClassifier:
    """Avoid changing a displayed label because of one noisy frame."""

    def __init__(self, window_size=10, min_votes=7):
        self.history = deque(maxlen=window_size)
        self.min_votes = min_votes
        self.current_label = "Initializing"

    def update(self, new_label):
        if new_label is None:
            return self.current_label

        self.history.append(new_label)

        if len(self.history) < self.history.maxlen:
            return self.current_label

        label, count = max(
            ((item, self.history.count(item))
             for item in set(self.history)),
            key=lambda item: item[1]
        )

        if count >= self.min_votes:
            self.current_label = label

        return self.current_label
