import numpy as np
from sklearn.model_selection import train_test_split

NUM_ITEMS = 30
HIST_DAYS = 28
FRC_DAYS = 2
RARE_DAYS_THRESHOLD = 3
MAX_SKIPS = 2


class ItemUserImage:
    def __init__(self):
        pass

    def collect_stats(self, order_json):
        """Collects users, items and dates statistics from data dict"""
        users = set()
        items = dict()
        dates = set()
        for order in order_json:
            users.add(order['userID'])
            dates.add(order['orderDate'])
            for pos in order['order']:
                category = pos['itemID']
                if items.get(category, None) is None:
                    items[category] = 1
                else:
                    items[category] += 1
        stats = {}
        items = [(k, v) for k, v in items.items()]
        stats['users'] = {user: i for i, user in enumerate(users)}
        stats['dates'] = {date: i for i, date in enumerate(sorted(dates))}
        stats['items'] = {item[0]: i for i, item in enumerate(sorted(items, key=lambda x: x[1], reverse=True))}
        return stats

    def form_data(self, order_json):
        """
        Transforms data dict to numpy array with with shape (|U| x |I| x |T|)
        where data_array[u, i, t] = 1 if user u bought item i at time t
        """
        stats = self.collect_stats(order_json)
        data_array = np.zeros(shape=(len(stats['users']), len(stats['dates']), len(stats['items'])))
        for order in order_json:
            user_id = stats['users'][order['userID']]
            date_id = stats['dates'][order['orderDate']]
            for pos in order['order']:
                item_id = stats['items'][pos['itemID']]
                data_array[user_id, date_id, item_id] = 1
        return data_array.reshape((*data_array.shape, -1))

    def reshape_conv(self, X):
        """Adds additional dimension (image depth) to data"""
        return X.reshape((*X.shape, -1))

    def get_first_visit_day(self, arr):
        """Returns indices of first nonzero elements along time axis.
        If no such element exists, returns number = 1 + number of steps in time axis"""
        first_visit_day = np.argmax(arr.max(axis=2), axis=1)
        no_visits = 1 - arr.max(axis=2).max(axis=1)
        timesteps_num = arr.shape[1]
        return np.maximum(first_visit_day, no_visits * timesteps_num).astype(int)

    def encode_array(self, arr):
        """One-hot encoding of integer array"""
        max_label = int(max(arr))
        bin_array = np.zeros((arr.size, max_label+1), dtype=int)
        bin_array[np.arange(arr.size), arr] = 1
        return bin_array

    def is_regular(self, img):
        visits = np.flatnonzero(img[:, :, 0].sum(axis=1))
        diff = np.diff(visits)
        m = np.median(diff)
        too_rare = (HIST_DAYS / max(len(visits), 1) > RARE_DAYS_THRESHOLD)
        too_many_skips = np.count_nonzero(diff > m * 2) > MAX_SKIPS
        if too_rare or too_many_skips:
            return False
        else:
            return True

    def sample_data(self, formed_data, ratio=0.8):
        """Returns x_train, x_test, y_train, y_test"""
        reg_x = []
        reg_y = []
        irreg_x = []
        irreg_y = []
        for user in formed_data:
            for timestep in range(user.shape[0] - HIST_DAYS - FRC_DAYS):
                x = user[timestep:timestep + HIST_DAYS, :NUM_ITEMS, :]
                y = user[timestep + HIST_DAYS:timestep + HIST_DAYS + FRC_DAYS, :, 0].sum() == 0
                if self.is_regular(x):
                    reg_x.append(x)
                    reg_y.append(y)
                else:
                    irreg_x.append(x)
                    irreg_y.append(y)
        xr_train, xr_test, yr_train, yr_test = train_test_split(np.array(reg_x), np.array(reg_y), train_size=ratio)
        xi_train, xi_test, yi_train, yi_test = train_test_split(np.array(irreg_x), np.array(irreg_y), train_size=ratio)
        return (xr_train, xr_test, yr_train, yr_test), (xi_train, xi_test, yi_train, yi_test)

    def binarize_label(self, arr, thresholds=(2, 7)):
        if len(thresholds) == 0:
            return self.encode_array(arr)
        label = np.zeros(arr.size, dtype=int)
        for i, t in enumerate(thresholds):
            label[arr >= t] = i+1
        if len(thresholds) > 1:
            return self.encode_array(label)
        else:
            return label

    def transform(self, data):
        dataset = self.form_data(data)
        sampled = self.sample_data(dataset)
        return sampled


