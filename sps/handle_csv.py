import csv
import os
from datetime import datetime
import json
from sps.config import Config


def _build_layer1_qrange_rules():
    """Layer-1 rules equivalent to q = floor(pixel * Q_RANGE / 256)."""
    rules = []

    for level in range(Config.Q_RANGE - 1, 0, -1):
        lower = (level * 256 + Config.Q_RANGE - 1) // Config.Q_RANGE #change 256 into the maximum range of the current image
        rules.append(f"[1,{lower},{lower},{level},0]")

    rules.append("[1,1,0,0,0]")
    return rules


def _with_negative_forgetting(rules):
    """Ensure anti-spike forgetting rule is present in generated CSV rules."""
    negative_rule = "[-1,-1,0,0,0]"
    if negative_rule in rules:
        return rules
    return rules + [negative_rule]


def _build_layer2_rules(k_index):
    """Layer-2 forwarding rules for positive charge range only."""
    rules = []

    for i in range(Config.K_RANGE[k_index][1], 0, -1):
        rules.append(f"[0,{i},{i},{i},0]")

    return _with_negative_forgetting(rules)


def SNPS_csv():
    """Generate the SN P system to replicate the cnn structure"""
    os.makedirs("csv", exist_ok=True)
    with open("csv/" + Config.CSV_NAME, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "initial_charge", "output_targets", "neuron_type", "rules"])

        # Layer 1: Input a 28x28 grayscale image
        l1_firing_rules = _build_layer1_qrange_rules()
        for neuron_id in range(Config.NEURONS_L1):
            i_row = neuron_id // Config.IMG_SHAPE
            i_col = neuron_id % Config.IMG_SHAPE
            output_targets = []

            for k_index, kernel in enumerate(Config.KERNELS):
                layer2_offset = Config.NEURONS_L1 + k_index * Config.NEURONS_FEATURE

                for ki in range(Config.KERNEL_SHAPE):
                    for kj in range(Config.KERNEL_SHAPE):
                        o_row = i_row - ki
                        o_col = i_col - kj

                        if 0 <= o_row < Config.SHAPE_FEATURE and 0 <= o_col < Config.SHAPE_FEATURE:
                            output_idx = o_row * Config.SHAPE_FEATURE + o_col
                            target_id = layer2_offset + output_idx
                            weight = kernel[ki][kj]
                            if weight == 1:
                                output_targets.append(target_id)
                            elif weight == -1:
                                output_targets.append(-target_id)

            writer.writerow([
                neuron_id,                     # id
                0,                             # initial_charge
                str(output_targets),           # output_targets
                0,                             # neuron_type
                *l1_firing_rules               # firing rules
            ])

        # Layer 2: Accumulate spikes from the kernels and extract features
        for k_index in range(len(Config.KERNELS)):
            l2_firing_rules = _build_layer2_rules(k_index)

            layer2_offset = Config.NEURONS_L1 + k_index * Config.NEURONS_FEATURE

            for i in range(Config.NEURONS_FEATURE):
                output_targets = [] # Target definition
                j = ((i // Config.SHAPE_FEATURE) // Config.POOLING_SIZE) * Config.SHAPE_POOL + ((i % Config.SHAPE_FEATURE) // Config.POOLING_SIZE) # position in next pooling layer
                output_targets.append(Config.NEURONS_L1 + Config.NEURONS_L2 + (k_index * Config.NEURONS_POOL) + j)
                writer.writerow([
                    layer2_offset + i,       # id
                    0,                       # initial_charge
                    str(output_targets),     # output_targets
                    1,                       # neuron_type
                    *l2_firing_rules         # Send all the spikes
                ])

        # Layer 3: Apply an average pooling on previous layer
        for k_index in range(Config.KERNEL_NUMBER):
            layer3_offset = Config.NEURONS_L1 + Config.NEURONS_L2 + k_index * Config.NEURONS_POOL
            for i in range(Config.NEURONS_POOL):
                writer.writerow([
                    layer3_offset + i,       # id
                    0,                       # initial_charge
                    "[]",                    # output_targets
                    2,                       # neuron_type
                    *_with_negative_forgetting(["[1,1,0,0,0]"])  # Send spikes + anti-spike forgetting
                ])



def ensemble_csv(svm_q, logreg_q, svm_imp, logreg_imp):
    """Generate the SN P system with the ensemble of two models"""
    os.makedirs("csv", exist_ok=True)
    with open("csv/" + Config.CSV_ENS_NAME, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "initial_charge", "output_targets", "neuron_type", "rules"])

        # Layer 1: Input an image and send the corresponding spike
        l1_firing_rules = _build_layer1_qrange_rules()
        for neuron_id in range(Config.NEURONS_L1):
            i_row = neuron_id // Config.IMG_SHAPE
            i_col = neuron_id % Config.IMG_SHAPE
            output_targets = []

            for k_index, kernel in enumerate(Config.KERNELS):
                layer2_offset = Config.NEURONS_L1 + k_index * Config.NEURONS_FEATURE

                for ki in range(Config.KERNEL_SHAPE):
                    for kj in range(Config.KERNEL_SHAPE):
                        o_row = i_row - ki
                        o_col = i_col - kj

                        if 0 <= o_row < Config.SHAPE_FEATURE and 0 <= o_col < Config.SHAPE_FEATURE:
                            output_idx = o_row * Config.SHAPE_FEATURE + o_col
                            target_id = layer2_offset + output_idx
                            weight = kernel[ki][kj]
                            if weight == 1:
                                output_targets.append(target_id)
                            elif weight == -1:
                                output_targets.append(-target_id)

            writer.writerow([
                neuron_id,                     # id
                0,                             # initial_charge
                str(output_targets),           # output_targets
                0,                             # neuron_type
                *l1_firing_rules                  # firing rules
            ])

        # Layer 2: Accumulate spikes from the kernels and extract features
        for k_index in range(len(Config.KERNELS)):
            l2_firing_rules = _build_layer2_rules(k_index)

            layer2_offset = Config.NEURONS_L1 + k_index * Config.NEURONS_FEATURE

            for i in range(Config.NEURONS_FEATURE):
                output_targets = [] # Target definition
                j = ((i // Config.SHAPE_FEATURE) // Config.POOLING_SIZE) * Config.SHAPE_POOL + ((i % Config.SHAPE_FEATURE) // Config.POOLING_SIZE) # position in next pooling layer
                first_target = Config.NEURONS_L1 + Config.NEURONS_L2 + (k_index * Config.NEURONS_POOL) + j
                second_target = first_target + Config.NEURONS_L3
                output_targets.append(first_target)
                output_targets.append(second_target)

                writer.writerow([
                    layer2_offset + i,       # id
                    0,                       # initial_charge
                    str(output_targets),     # output_targets
                    1,                       # neuron_type
                    *l2_firing_rules         # Send all the spikes
                ])

        # Layer 3 - average pooling and apply two different synapses matrices
        pool_offset = Config.NEURONS_L1 + Config.NEURONS_L2
        output_offset = Config.NEURONS_L1 + Config.NEURONS_L2 + 2 * Config.NEURONS_L3
        rows_to_add = build_rows(pool_offset, svm_q, svm_imp) #First classification: svm
        for row in rows_to_add:
            writer.writerow(row)

        new_start = pool_offset + len(rows_to_add)
        rows_to_add = build_rows(new_start, logreg_q, logreg_imp) #Second classification: logreg
        for row in rows_to_add:
            writer.writerow(row)

        for j in range(Config.CLASSES):
            output_rules = _with_negative_forgetting(["[1,1,0,0,0]"])
            row = [
                output_offset + j,   # id
                0,             # initial charge
                "[]",          # no output
                2,             # neuron type (accumulator/output)
                *output_rules   # send spikes + anti-spike forgetting
            ]
            writer.writerow(row)

    return "csv/" + Config.CSV_ENS_NAME


def build_rows(start_offset, q, multipliers=None):
    #create a layer 3 block using weights and multipliers as input
    output_offset = Config.NEURONS_L1 + Config.NEURONS_L2 + 2 * Config.NEURONS_L3
    new_rows = []

    for i in range(Config.NEURONS_L3):
        new_targets = []
        for j in range(Config.CLASSES):
            weight = q[i, j]
            j = j + output_offset

            if weight == 1:
                new_targets.append(j)
            elif weight == -1:
                new_targets.append(-j)

        new_rules = []
        for out_spikes in range(Config.K_RANGE[0][1], 0, -1):
            k = Config.POOLING_SIZE ** 2 * out_spikes
            multiplied = int(out_spikes * multipliers[i]) if multipliers is not None else out_spikes
            new_rules.append(str([1, k, k, multiplied, 0]))

        new_rules = _with_negative_forgetting(new_rules)

        row = [str(start_offset + i), "0", str(new_targets), "1"] + new_rules
        new_rows.append(row)

    return new_rows

def extend_csv(file_path, q, q_name, multipliers):
    # create a new version of the csv with new output_targets and rules based on q and multipliers
    base, ext = os.path.splitext(file_path)
    new_file_path = f"{base}_{q_name}{ext}"

    with open(file_path, newline='') as f:
        rows = list(csv.reader(f))
    output_offset = Config.NEURONS_L1 + Config.NEURONS_L2 + Config.NEURONS_L3
    pool_offset = Config.NEURONS_L1 + Config.NEURONS_L2
    n_classes = min(Config.CLASSES, q.shape[1])

    for i in range(Config.NEURONS_L3):
        row = rows[i+pool_offset+1]
        # new output targets based on q
        new_targets = []
        for j in range(n_classes):

            weight = q[i, j]
            j = j + output_offset

            if weight == 1:
                new_targets.append(j)
            elif weight == -1:
                new_targets.append(-j)
            if len(row) < 3:
                print(f"Warning: row {i} too short:", row)
                row += [''] * (3 - len(row))
        row[2] = str(new_targets)

        #New firing rules
        new_rules = []
        for out_spikes in range(Config.K_RANGE[0][1], 0, -1):  # from 48 to 1
            k = Config.POOLING_SIZE ** 2 * out_spikes  # 48*4, 47*4, ..., 1*4
            multiplied = int(out_spikes * multipliers[i]) if multipliers is not None else out_spikes #rules tuning using importance
            new_rules.append(str([1, k, k, multiplied, 0]))

        new_rules = _with_negative_forgetting(new_rules)

        row[:] = row[:4] + new_rules

    #add new rows for classes's output neurons
    for j in range(Config.CLASSES):

        output_rules = _with_negative_forgetting(["[1,1,0,0,0]"])

        new_row = [
            output_offset + j - 1,   # id
            0,             # initial charge
            "[]",          # no output
            2,              # neuron type (accumulator/output)
            *output_rules   # send spikes + anti-spike forgetting
        ]

        rows.append(new_row)

    # write the new csv
    with open(new_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return new_file_path


def save_results(ensemble_accuracy, time):
    log_experiment(
        params={
            "train size": Config.TRAIN_SIZE,
            "test size": Config.TEST_SIZE,
            "q range": Config.Q_RANGE,
            "svm c": Config.SVM_C,
            "ternarize method": Config.TERNARIZE_METHOD,
            "importance method": Config.IMPORTANCE_METHOD,
            "discretize method": Config.DISCRETIZE_METHOD,
            "discretization range": Config.DISC_RANGE,
            "matrix sparsity": Config.M_SPARSITY,
            "matrix positive": Config.M_POSITIVE,
            "matrix threshold": Config.M_THRESHOLD,
            "database": Config.DATABASE,
            "kernel number": Config.KERNEL_NUMBER
        },
        metrics={
            "ensemble accuracy": ensemble_accuracy,
            "time": time
        }
    )


def log_experiment(csv_path="csv/results.csv", params=None, metrics=None):
    """
    log_experiment(csv_path="csv/results.csv", params=None, metrics=None)
    """
    file_exists = os.path.isfile(csv_path)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": json.dumps(params or {}),
        "metrics": json.dumps(metrics or {})
    }

    fieldnames = ["timestamp", "params", "metrics"]

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_accuracies_sparsity(
        csv_path="csv/accuracies_sparsity.csv",
        accuracies=None,   # list of 10 values
        train_size=None,
        q_range=None,
        sparsity=None,
        positive=None,
        negative=None
):
    file_exists = os.path.isfile(csv_path)

    accuracies = accuracies or [None] * 10

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_size": train_size,
        "q_range": q_range,
        "sparsity": sparsity,
        "positive": positive,
        "negative": negative,
    }

    # add accuracy columns
    for i in range(10):
        row[f"acc_{i}"] = accuracies[i]

    fieldnames = (
            ["timestamp", "train_size", "q_range", "sparsity", "positive", "negative"] +
            [f"acc_{i}" for i in range(10)]
    )

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)