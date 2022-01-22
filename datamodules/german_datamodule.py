from datamodule import Datamodule

class GermanDatamodule(Datamodule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__('data.npz', 36, -0.6702800625998365, data_dir, batch_size)


if __name__ == '__main__':
    print(GermanDatamodule('./data', 1).get_train_dataset()[0])
