import glob


def get_all_models(base_path):
    return sorted(glob.glob("{}/*.chk".format(base_path)))


def find_best_model(base_path):
    path_ls = get_all_models(base_path)
    if len(path_ls) == 1:
        path = path_ls[0]
    else:
        path = sorted(path_ls, key=lambda p: -int(p.split("_")[-1].split(".")[0]))[0]
    return path
