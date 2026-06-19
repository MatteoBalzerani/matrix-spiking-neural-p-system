import numpy as np
import torch
from sps.snp_system import SNPSystem
from sps.config import Config
from sps.m_snp_pytorch_exact_GPU_and_CPU import MSNPSystemExactGPU


class MatrixExecutor:

    # This class is responsible for translating a SNPSystem from the sps.snp_system format to the MSNPSystem format
    @staticmethod
    def translate_to_matrix(SNPSystem,device="cpu",debugMode=""):
        neurons = SNPSystem.neurons
        neurons_num = len(neurons)
        rule_num = sum(len(neuron.transf_rules) for neuron in neurons)

        deterministic = SNPSystem.deterministic
        max_steps = SNPSystem.max_steps

        # Initialize the vectors and matrices
        configurationVector = np.zeros(neurons_num, dtype=int)
        spikingVector = np.zeros((rule_num,), dtype=int)
        #spikingTransitionMatrix = np.zeros((rule_num, neurons_num), dtype=int)  # as explained in the paper
        #synapsesMatrix = np.zeros((rule_num, neurons_num), dtype=int)
        # I implemented these vectors in order to make the system executable
        ruleVector = np.zeros(rule_num, dtype=int)  # exact E goes in mod, div is for all rules in this implementation
        applyingRuleVector = np.zeros((rule_num,), dtype=int)  # which neuron each rule applies to

        # Build sMpi directly as sparse COO — never allocate dense matrices
        sparse_rows = []   # rule index
        sparse_cols = []   # neuron index
        sparse_vals = []   # value (spikingTransitionMatrix * synapsesMatrix, but synapses=1 wherever there's an entry)


        rule_idx = 0
        input_neurons = []  # index of input neurons, to which the spike train will be applied
        output_neurons = []  # index of output neurons, to which the output will be read from

        for neuron in neurons:
            if neuron.neuron_type == 0:  # if it's an input neuron, add it to the list of input neurons
                input_neurons.append(neuron.nid)
            elif neuron.neuron_type == 2:  # if it's an output neuron, add it to the list of output neurons
                output_neurons.append(neuron.nid)

            configurationVector[neuron.nid] = neuron.charge

            for rule in neuron.transf_rules:
                # Source neuron entry: -rule.source (same as spikingTransitionMatrix * synapsesMatrix)
                sparse_rows.append(rule_idx)
                sparse_cols.append(neuron.nid)
                sparse_vals.append(-rule.source)

                #spikingTransitionMatrix[rule_idx, neuron.nid] = -rule.source
                #synapsesMatrix[rule_idx, neuron.nid] = 1

                for target in neuron.targets:
                    actual_target = abs(target)
                    spike_sign = 1 if target > 0 else -1
                    sparse_rows.append(rule_idx)
                    sparse_cols.append(actual_target)
                    sparse_vals.append(rule.target * spike_sign)

                    #actual_target = abs(target)  # indice reale del neurone
                    #spike_sign = 1 if target > 0 else -1  # segno dello spike
                    #spikingTransitionMatrix[rule_idx, actual_target] = rule.target * spike_sign
                    #synapsesMatrix[rule_idx, actual_target] = 1


                    #spikingTransitionMatrix[rule_idx, target] = rule.target
                    #synapsesMatrix[rule_idx, target] = 1 if target > 0 else -1

                ruleVector[rule_idx] = rule.mod  # exact goes in mod - assume div = 0

                applyingRuleVector[rule_idx] = neuron.nid
                rule_idx += 1

        # Build sparse tensor directly — peak memory is just the nonzero values
        indices = torch.tensor([sparse_rows, sparse_cols], dtype=torch.long)
        values = torch.tensor(sparse_vals, dtype=torch.float32)
        sMpi_sparse = torch.sparse_coo_tensor(indices, values, (rule_num, neurons_num))
        sMpi_sparse = sMpi_sparse.coalesce()  # merge duplicate indices by summing

        # Define single_spike_train along the Config Mode
        single_spike_train = None
        if Config.MODE != "CNN":
            single_spike_train = SNPSystem.spike_train

        # Create and return instance of MSNPSystemExactGPU (PyTorch version)
        return MSNPSystemExactGPU(
            configurationVector=configurationVector,
            spikingVector=spikingVector,
            sMpi_sparse=sMpi_sparse,          # pass sparse directly
            ruleVector=ruleVector,
            max_steps=max_steps,
            deterministic=deterministic,
            single_spike_train=single_spike_train,
            input_neurons=input_neurons,
            output_neurons=output_neurons,
            applyingRuleVector=applyingRuleVector,
            device=device,
            testsize=SNPSystem.input_len,
            debugMode=debugMode
        )