from scipy.spatial import distance_matrix
from argparse import ArgumentParser
import numpy as np
import logging
import math
import time
import tqdm
from collections import defaultdict
import pandas as pd
import joblib
import glob
import yaml
import os


logger = logging.getLogger(__name__)


scale_x = 2.96e-06
scale_y = 2.96e-06
scale_z = 1.00e-06
scale_d = 2.96e-06


scales = [scale_x, scale_y, scale_z, scale_d]


class Threshold:

    def __init_(self):
        self.n = None

    def load_dist_matrix(self, h_idx, coordinates):

        vol = np.array(coordinates[h_idx], dtype=float)

        """apply the relevant scale transformation"""
        vol[:, 0] *= scale_x
        vol[:, 1] *= scale_y
        vol[:, 2] *= scale_z
        vol[:, 3] *= scale_d

        self.vol = vol
        self.dist_matrix = distance_matrix(vol, vol)
        self.n = self.dist_matrix.shape[0]

    def search(self, threshold):
        results = {}
        for label in range(self.n):
            """Find experiments at or below the threshold"""
            members = np.where(self.dist_matrix[label] <= threshold)[0]
            results[label] = [x for x in members if x != label]
        results = sorted([
            [len(x), i, x] for (i, x) in results.items()
        ])
        return results  # sorted(results, reverse=True)

    def Cluster(self, threshold):
        """Use Leader clustering"""
        true_singletons = []
        false_singletons = []
        clusters = []
        seen = set()
        for (size, experiment, members) in self.search(threshold):
            if experiment in seen:
                """Can't use a centroid which is already assigned"""
                continue
            seen.add(experiment)
            """Figure out which ones haven't yet been assigned"""
            unassigned = set(members) - seen
            if not unassigned:
                false_singletons.append(experiment)
                continue
            """this is a new cluster"""
            clusters.append((experiment, unassigned))
            seen.update(unassigned)

        seen = []
        for a, b in clusters:
            seen.append(a)
            for c in b:
                if c not in seen:
                    seen.append(c)

        not_clustered = set(range(self.n)) - set(seen)
        return sorted(clusters, key=lambda x: -len(x[1])), list(not_clustered)


def diameter_average(coors, centroid, clusters):
    centroid = [coors[centroid]]
    centroid += [coors[x] for x in clusters]
    centroid = np.array(centroid).astype(float)
    centroid = np.average(centroid, axis=0)
    return centroid


def distance(x, y):
    return math.sqrt(sum([(scales[i] * x[i] - scales[i] * y[i])**2 for i, _ in enumerate(x)]))


def create_table(distance_threshold=0.001,
                 true_coordinates=None,
                 pred_coordinates=None,
                 match=True):

    mapping_table = defaultdict(list)
    mapping_table["rmse"] = {}
    df_table = defaultdict(list)

    holo_indices = list(true_coordinates.keys()) if match else list(
        pred_coordinates.keys())
    for h_idx in tqdm.tqdm(sorted(holo_indices)):

        logger.info(
            f"Starting hologram {h_idx}  ...")

        if match:
            if h_idx not in pred_coordinates:
                for p in true_coordinates[h_idx]:
                    mapping_table[h_idx].append([" ".join(map(str, p)), None])
                    x, y, z, d = list(map(float, p))
                    df_table["h"].append(h_idx)
                    df_table["x_t"].append(x)
                    df_table["y_t"].append(y)
                    df_table["z_t"].append(z)
                    df_table["d_t"].append(d)
                    df_table["x_p"].append(np.nan)
                    df_table["y_p"].append(np.nan)
                    df_table["z_p"].append(np.nan)
                    df_table["d_p"].append(np.nan)
                    df_table["rmse"].append(np.nan)
                continue

            if len(pred_coordinates[h_idx]) == 0:
                for p in true_coordinates[h_idx]:
                    mapping_table[h_idx].append([" ".join(map(str, p)), None])
                    x, y, z, d = list(map(float, p))
                    df_table["h"].append(h_idx)
                    df_table["x_t"].append(x)
                    df_table["y_t"].append(y)
                    df_table["z_t"].append(z)
                    df_table["d_t"].append(d)
                    df_table["x_p"].append(np.nan)
                    df_table["y_p"].append(np.nan)
                    df_table["z_p"].append(np.nan)
                    df_table["d_p"].append(np.nan)
                    df_table["rmse"].append(np.nan)
                continue

        """Cluster particles using the distance matrix and threshold"""
        start_time = time.time()
        t = Threshold()
        t.load_dist_matrix(h_idx, pred_coordinates)
        clusters, unassigned = t.Cluster(distance_threshold)

        """Create numpy arrays from the centroids/unassigned"""
        pred_r_centroids = np.array([diameter_average(
            pred_coordinates[h_idx], centroid, members) for centroid, members in clusters]).astype(float)
        pred_r_not_matched = np.array(
            [pred_coordinates[h_idx][idx] for idx in unassigned]).astype(float)
        if pred_r_centroids.shape[0] > 0:
            if pred_r_not_matched.shape[0] > 0:
                pred_r = np.concatenate([pred_r_centroids, pred_r_not_matched])
            else:
                pred_r = pred_r_centroids
        else:
            pred_r = pred_r_not_matched
        ctime = time.time() - start_time

        logger.info(
            f"... clustering completed in {ctime} s, {len(clusters)} clusters, {len(unassigned)} unassigned, {pred_r.shape[0]} total")

        if not match:
            mapping_table[h_idx] = [list(x) for x in pred_r]
            for coors in pred_r:
                x, y, z, d = list(coors)
                df_table["x"].append(x)
                df_table["y"].append(y)
                df_table["z"].append(z)
                df_table["d"].append(d)
                df_table["h"].append(h_idx)
            continue

        """
            Match the clustered particles against the true particles (if they exist)
        """

        true_r = np.array(true_coordinates[h_idx]).astype(float)

        """Compute the distance matrix b/t the two datasets --> pandas df"""
        start_time = time.time()
        result_dict = defaultdict(list)
        for k1, x in enumerate(pred_r):
            for k2, y in enumerate(true_r):
                error = distance(x, y)
                result_dict["pred_id"].append(k1)
                result_dict["true_id"].append(k2)
                result_dict["pred_coor"].append(
                    " ".join([str(xx) for xx in list(x)]))
                result_dict["true_coor"].append(
                    " ".join([str(yy) for yy in list(y)]))
                result_dict["error"].append(np.mean(np.abs(error)))
        df = pd.DataFrame(result_dict)

        """Add to the mapping table"""
        pred_seen = []
        true_seen = []
        error = []
        while True:
            c1 = df["true_id"].isin(true_seen)
            c2 = df["pred_id"].isin(pred_seen)
            c = c1 | c2
            if c.sum() == df.shape[0]:
                break
            smallest_error = df[~c]["error"] == min(df[~c]["error"])
            error.append(list(df[~c][smallest_error]["error"])[0])
            true_id = list(df[~c][smallest_error]["true_id"])[0]
            pred_id = list(df[~c][smallest_error]["pred_id"])[0]
            pred_seen.append(pred_id)
            true_seen.append(true_id)

            pred_n = list(df[~c][smallest_error]["pred_coor"])[0]
            true_n = list(df[~c][smallest_error]["true_coor"])[0]
            mapping_table[h_idx].append([true_n, pred_n])

        """Add matched to the table/dataframe"""
        for idx, (true, pred) in enumerate(mapping_table[h_idx]):
            df_table["h"].append(h_idx)
            x, y, z, d = list(map(float, true.split(" ")))
            df_table["x_t"].append(x)
            df_table["y_t"].append(y)
            df_table["z_t"].append(z)
            df_table["d_t"].append(d)
            x, y, z, d = list(map(float, pred.split(" ")))
            df_table["x_p"].append(x)
            df_table["y_p"].append(y)
            df_table["z_p"].append(z)
            df_table["d_p"].append(d)
            df_table["rmse"].append(error[idx])
        n_match = len(mapping_table[h_idx])

        """Add non-matched to the table/dataframe"""

        true_unmatched = list(
            set(df["true_coor"].unique()) - set([x[0] for x in mapping_table[h_idx]]))
        pred_unmatched = list(
            set(df["pred_coor"].unique()) - set([x[1] for x in mapping_table[h_idx]]))

        for p in true_unmatched:
            mapping_table[h_idx].append([p, None])
            x, y, z, d = list(map(float, p.split(" ")))
            df_table["h"].append(h_idx)
            df_table["x_t"].append(x)
            df_table["y_t"].append(y)
            df_table["z_t"].append(z)
            df_table["d_t"].append(d)
            df_table["x_p"].append(np.nan)
            df_table["y_p"].append(np.nan)
            df_table["z_p"].append(np.nan)
            df_table["d_p"].append(np.nan)
            df_table["rmse"].append(np.nan)

        for p in pred_unmatched:
            mapping_table[h_idx].append([None, p])
            x, y, z, d = list(map(float, p.split(" ")))
            df_table["h"].append(h_idx)
            df_table["x_p"].append(x)
            df_table["y_p"].append(y)
            df_table["z_p"].append(z)
            df_table["d_p"].append(d)
            df_table["x_t"].append(np.nan)
            df_table["y_t"].append(np.nan)
            df_table["z_t"].append(np.nan)
            df_table["d_t"].append(np.nan)
            df_table["rmse"].append(np.nan)

        mtime = time.time() - start_time
        logger.info(
            f"... matched {n_match} particles in {mtime} s. RMSE = {np.mean(error)}")

        mapping_table["rmse"][h_idx] = error

    return mapping_table, df_table


if __name__ == "__main__":

    description = "1. Cluster predictions using (x,y,x,d) predictions in N planes from M workers\n"
    description += "2. Pair matched particles against true or the standard method predictions"

    parser = ArgumentParser(
        description=description
    )
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs."
    )
    parser.add_argument(
        "-m",
        dest="match",
        type=str,
        default=True,
        help="Whether to match predictions against truth or standard method values."
    )

    args_dict = vars(parser.parse_args())
    config_file = args_dict.pop("model_config")
    match = bool(int(args_dict.pop("match")))

    with open(config_file) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    n_nodes = conf["inference"]["n_nodes"]
    n_gpus = conf["inference"]["gpus_per_node"]
    threads_per_gpu = conf["inference"]["threads_per_gpu"]
    workers = int(n_nodes * n_gpus * threads_per_gpu)

    save_loc = conf["save_loc"]
    inf_save_loc = conf["inference"]["data_set"]["name"]
    path_to_preds = os.path.join(save_loc, inf_save_loc, "propagated")
    distance_threshold = conf["inference"]["distance_threshold"]

    ############################################################
    # Initialize logger to stream to stdout
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Save the log file
    logger_name = os.path.join(os.path.join(
        save_loc, f"{inf_save_loc}/clustering_log.txt"))
    fh = logging.FileHandler(logger_name,
                             mode="w",
                             encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root.addHandler(fh)
    ############################################################
    """Obtain the pred and true files produced by the workers"""
    fns = glob.glob(os.path.join(path_to_preds, "*txt"))

    preds = [x for x in fns if "pred" in x.split("/")[-1]]

    logger.info(f"The number of workers used during inference was {workers}")
    logger.info(f"There {len(preds)} files containing predicted coordinates")

    if len(preds) != workers:
        logger.warning(
            "The number of files with coordinates is not equal to the number of workers in the config file")

    pred_coordinates = defaultdict(list)
    for fn in preds:
        with open(fn, "r") as fid:
            for line in fid.readlines():
                h, x, y, z, d = list(map(int, line.split(" ")))
                pred_coordinates[h].append([x, y, z, d])

    if match:
        truth = [x for x in fns if "true" in x.split("/")[-1]]
        true_coordinates = defaultdict(list)
        for fn in truth:
            with open(fn, "r") as fid:
                for line in fid.readlines():
                    h, x, y, z, d = list(map(int, line.split(" ")))
                    true_coordinates[h].append([x, y, z, d])
    else:
        true_coordinates = None

    coors_table, df_table = create_table(
        distance_threshold=distance_threshold,
        true_coordinates=true_coordinates,
        pred_coordinates=pred_coordinates,
        match=match
    )

    """Save the table as a dictionary"""
    save_fn = os.path.join(save_loc,  inf_save_loc,
                           f"prediction_table_{str(distance_threshold)}.pkl")
    with open(save_fn, "wb") as fid:
        joblib.dump(coors_table, fid)

    """Save the table to csv"""
    df_table = pd.DataFrame.from_dict(df_table)
    df_table.to_csv(os.path.join(
        save_loc, inf_save_loc, f"prediction_table_{str(distance_threshold)}.csv"))
