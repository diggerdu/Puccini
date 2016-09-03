import os
import os.path
import numpy as np



class audio_data():
    def __init__(self):
        nega_path = "../nega"
        posi_path = "../posi"

        nega_data = np.zeros((1,))
        for file in os.listdir(nega_path):
            tmp = np.load(nega_path + '/' + file)
            if nega_data.shape[0] == 1:
                nega_data = tmp
            else:
                nega_data = np.vstack((nega_data, tmp))

        posi_data = np.zeros((1,))
        for file in os.listdir(posi_path):
            tmp = np.load(posi_path + '/' + file)
            if posi_data.shape[0] == 1:
                posi_data = tmp
            else:
                posi_data = np.vstack((posi_data, tmp))

        print ("positive data shape:", posi_data.shape)
        print ("negative_data shape:", nega_data.shape)
        self.data = np.vstack((posi_data, nega_data))
        self.label = np.hstack((np.ones(posi_data.shape[0]), np.zeros(nega_data.shape[0])))
    def next_batch(self, batch_size, prev = 0, sub = 0):
        batch_idx = np.random.choice(np.arange(prev, self.data.shape[0] - sub, 1), batch_size)
        batch_x = np.array([self.data[idx - prev : idx + sub + 1] for idx in batch_idx])
        batch_y = self.label[batch_idx]
        return batch_x, batch_y

if __name__ == "__main__":
    input_data = audio_data()
    print(input_data.next_batch(10,1,1)[0].shape)