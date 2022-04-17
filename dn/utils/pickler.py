import pickle


def load_image_data_pickle(path):
    from dn.data import ImageData, MyDS

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data
