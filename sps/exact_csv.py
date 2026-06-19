import csv
import os

from sps.config import Config
from sps.handle_csv import _with_negative_forgetting, _build_layer2_rules


'''Number of exact rules:
* layer 1 : 784 neuroni * 255 regole = 200k
* layer 2 : 5408 neuroni * (q range-1)*6 regole = 32k * q range
* layer 3 : 1532 neuroni * (q range-1)*24 regole = 32k * q range
---> total 200k + 64k * q range
'''

def SNPS_exact_csv():
    """Generate the SN P system with exact rules"""
    os.makedirs("csv", exist_ok=True)
    with open("csv/" + Config.CSV_EXACT_NAME, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "initial_charge", "output_targets", "neuron_type", "rules"])

        # Layer 1: Input a 28x28 grayscale image
        l1_firing_rules = _build_layer1_exact_rules()
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
            l2_firing_rules = _build_layer2_exact_rules(k_index)

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
            k_range_max = Config.K_RANGE[k_index][1]
            all_rules = _build_layer3_exact_rules(k_range_max, None)

            for i in range(Config.NEURONS_POOL):
                global_i = k_index * Config.NEURONS_POOL + i
                writer.writerow([
                    layer3_offset + i,       # id
                    0,                       # initial_charge
                    "[]",                    # output_targets
                    2,                       # neuron_type
                    *all_rules[global_i]  # Send spikes + anti-spike forgetting
                ])


def ensemble_exact_csv(svm_q, logreg_q, svm_imp, logreg_imp):
    """Generate the SN P system with the ensemble of two models"""
    os.makedirs("csv", exist_ok=True)
    with open("csv/" + Config.CSV_EXACT_ENS_NAME, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "initial_charge", "output_targets", "neuron_type", "rules"])

        # Layer 1: Input an image and send the corresponding spike
        l1_firing_rules = _build_layer1_exact_rules()
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
            l2_firing_rules = _build_layer2_exact_rules(k_index)

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

    return "csv/" + Config.CSV_EXACT_ENS_NAME




def build_rows(start_offset, q, multipliers=None):
    output_offset = Config.NEURONS_L1 + Config.NEURONS_L2 + 2 * Config.NEURONS_L3
    new_rows = []

    # Pre-compute exact rules for every neuron in this L3 block
    k_range_max = Config.K_RANGE[0][1]
    all_rules = _build_layer3_exact_rules(k_range_max, multipliers)

    for i in range(Config.NEURONS_L3):
        new_targets = []
        for j in range(Config.CLASSES):
            weight = q[i, j]
            target = j + output_offset
            if weight == 1:
                new_targets.append(target)
            elif weight == -1:
                new_targets.append(-target)

        row = [str(start_offset + i), "0", str(new_targets), "1"] + all_rules[i]
        new_rows.append(row)

    return new_rows


def extend_csv(file_path, q, q_name, multipliers):
    base, ext = os.path.splitext(file_path)
    new_file_path = f"{base}_{q_name}{ext}"

    with open(file_path, newline='') as f:
        rows = list(csv.reader(f))

    output_offset = Config.NEURONS_L1 + Config.NEURONS_L2 + Config.NEURONS_L3
    pool_offset   = Config.NEURONS_L1 + Config.NEURONS_L2
    n_classes     = min(Config.CLASSES, q.shape[1])

    k_range_max = Config.K_RANGE[0][1]
    all_rules   = _build_layer3_exact_rules(k_range_max, multipliers)

    for i in range(Config.NEURONS_L3):
        row = rows[i + pool_offset + 1]

        new_targets = []
        for j in range(n_classes):
            weight = q[i, j]
            target = j + output_offset
            if weight == 1:
                new_targets.append(target)
            elif weight == -1:
                new_targets.append(-target)
            if len(row) < 3:
                row += [''] * (3 - len(row))

        row[2] = str(new_targets)
        row[:] = row[:4] + all_rules[i]

    for j in range(Config.CLASSES):
        output_rules = ["[0,1,1,0,0]"]
        new_row = [
            output_offset + j - 1, 0, "[]", 2, *output_rules
        ]
        rows.append(new_row)

    with open(new_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return new_file_path


def _build_layer1_exact_rules():
    rules = []

    thresholds = []
    for level in range(Config.Q_RANGE - 1, 0, -1):
        lower = (level * 256 + Config.Q_RANGE - 1) // Config.Q_RANGE
        thresholds.append((lower, level))

    for idx, (lower, level) in enumerate(thresholds):
        upper = 255 if idx == 0 else thresholds[idx - 1][0] - 1
        for x in range(upper, lower - 1, -1):
            rules.append(f"[0,{x},{x},{level},0]")

    # Fill remaining values down to 1 with level 0
    lowest_lower = thresholds[-1][0]
    for x in range(lowest_lower - 1, 0, -1):
        rules.append(f"[0,{x},{x},0,0]")

    return rules


def _build_layer3_exact_rules(k_range_max, multipliers_vec):
    result = []

    for i in range(Config.NEURONS_L3):
        rules = []
        mult_i = multipliers_vec[i] if multipliers_vec is not None else 1

        thresholds = []
        for out_spikes in range(k_range_max, 0, -1):
            k = Config.POOLING_SIZE ** 2 * out_spikes
            multiplied = int(out_spikes * mult_i)
            thresholds.append((k, multiplied))

        for idx, (k_lower, multiplied) in enumerate(thresholds):
            k_upper = thresholds[idx - 1][0] - 1 if idx > 0 else k_lower
            for x in range(k_upper, k_lower - 1, -1):
                rules.append(f"[0,{x},{x},{multiplied},0]")

        # Fill remaining values down to 1 with 0 output
        lowest_k = thresholds[-1][0]
        for x in range(lowest_k - 1, 0, -1):
            rules.append(f"[0,{x},{x},0,0]")

        #max_negative = thresholds[0][0]  # symmetric: same as highest positive k
        #rules = _with_negative_forgetting_exact(rules, max_negative)
        result.append(rules)

    return result


def _with_negative_forgetting_exact(rules, max_negative):
    """Emit one exact rule per negative charge value from -1 down to -max_negative."""
    for x in range(1, max_negative + 1):
        rules.append(f"[0,{-x},{-x},0,0]")
    return rules

def _build_layer2_exact_rules(k_index):
    max_val = Config.K_RANGE[k_index][1]
    rules = []

    for i in range(max_val, 0, -1):
        rules.append(f"[0,{i},{i},{i},0]")

    # negative exact rules
    for i in range(1, max_val + 1):
        rules.append(f"[0,{-i},{-i},0,0]")

    return rules
