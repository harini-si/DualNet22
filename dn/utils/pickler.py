import pickle


def load_image_data_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data
