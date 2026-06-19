import time
import numpy as np
from sps.config import Config
from .p_neuron import PNeuron
from .spike_utils import SpikeEvent, TransformationRule, History
import csv

from .charge_tracker import ChargeTracker
from .system_measurers import TimerSNP

class SNPSystem:
    """Spiking Neural P System"""

    def __init__(self, input_len, max_steps, deterministic, debugMode=""):
        # Backward compatible constructor:
        # - (input_len, max_steps, deterministic)
        # - (max_delay, max_steps, input_type, output_type, deterministic)
        #if len(args) == 3:
        #    input_len, max_steps, deterministic = args
        #    max_delay = 5
        #    input_type = "images"
        #    output_type = "images"
        #elif len(args) == 5:
        #    max_delay, max_steps, input_type, output_type, deterministic = args
        #    input_len = max_steps
        #else:
        #    raise TypeError("SNPSystem expects 3 or 5 positional arguments")

        # TIMER FOR PERFORMANCE MEASUREMENT
        self._phase = debugMode  # "TRAIN" o "TEST"
        self._system_name = "SNPSystem_CPU"
        self.timer = TimerSNP(max_steps, debugMode+"_time_SNPSystem")

        PNeuron.reset_nid()
        self.input_len = input_len
        self.t_step = 0
        self.max_steps = max_steps
        self.deterministic = deterministic # can be true or false

        self.history = None

        self.max_delay = 5 # init circular future spiking events based on max_delay
        self.spike_events = [[] for _ in range(self.max_delay)]

        self.neurons = []
        self.spike_train = None
        self.output_neuron_ids = []

        # record output
        if Config.MODE == "generative":
            self.output = [] # time between two spikes in the output neuron
        elif Config.MODE == "CNN":
            self.pooling_image = np.zeros((Config.NEURONS_L3, input_len), dtype=int)
            self.labels = []
            self.correct = 0
            self.charge_map_prediction = np.zeros((Config.CLASSES, input_len), dtype=int) #values for the last layer of the cnn

        self.spike_fired = 0
        self.inhibition_fired = 0
        self.firing_applied = 0
        self.forgetting_applied = 0

        self.charge_tracker = None # I initialize the charge tracker 
    def init_history(self):
        """init tick history based on the system's neurons"""
        self.history = History(self.neurons)

    def start(self):
        """start sending and receiving spikes"""
        self.init_history()
        if getattr(Config, "TRACK_CHARGES", False):
            self.charge_tracker = ChargeTracker(
                filename=getattr(Config, "TRACK_FILENAME", "output_charges"),
                mode=getattr(Config, "TRACK_MODE", "step_by_step"),
                format=getattr(Config, "TRACK_FORMAT", "csv"),
                num_neurons=len(self.neurons),
            )
        else:
            self.charge_tracker = None
        # keep on ticking until output condition is met or max number of ticks is exceeded
        while True:
            if self.charge_tracker is not None:
                self.charge_tracker.record_charges(self.t_step, self.neurons) #takes the index of the current image and reads the charge of each neuron and saves it in self.history.
            if not self.tick():
                # calculate consumed energy
                """For more info, see latex document and chapter 5.5 of 'Beyond classification: directly training spiking
                neural networks for semantic segmentation' -> paper "https://arxiv.org/pdf/2110.07742" """
                total_synapses = sum(len(neuron.targets) for neuron in self.neurons)
                total_rules = sum(len(neuron.transf_rules) for neuron in self.neurons)
                w_energy = self.t_step * (total_synapses + total_rules * Config.WORST_REGEX)
                e_energy = int(self.spike_fired * Config.EXPECTED_SPIKE + (self.firing_applied + self.forgetting_applied) * Config.EXPECTED_REGEX)
                if self.charge_tracker is not None:
                    self.charge_tracker.finish()

                self.timer.export_step_times(self._system_name, "PerStep", self._phase)

                if Config.MODE == "generative":
                    print("Spike fired at time step", self.output[0], "and time step", self.output[1], ". The output is", self.output[1] - self.output[0])
                np.save("/tmp/charge_map_snp.npy", self.charge_map_prediction)
                print(f"Saved charge_map_snp: {self.charge_map_prediction.shape}")
                return w_energy, e_energy

    def tick(self):
        """at each time step, first evolve and then receive spikes, cant do both in the same step, refractory will prevent it"""
        
        self.timer.start_step(self.t_step)

        self.history.add_new_tick()

        any_rule_applied = False
        # evolve each neuron
        for neuron in self.neurons:
            used_rule = neuron.tick()
            if used_rule:
                any_rule_applied = True
            # fire event
            if used_rule and used_rule.target > 0:
                # Generate a firing event that will be received in the future, if it has delay
                self.spike_events[(self.t_step + used_rule.delay) % self.max_delay].append(SpikeEvent(neuron.nid, used_rule.target, neuron.targets))
                if neuron.neuron_type == 2 and Config.MODE == "generative": # output neuron
                    self.output.append(self.t_step)
            self.history.record_rule(neuron, used_rule)

        input_spike = False # check if there are more input for halting condition
        if Config.MODE == "CNN": # you have an array of images as input
            if self.spike_train.any() and self.t_step < len(self.spike_train):
                input_spike = True
                input_vector = self.spike_train[self.t_step].flatten() # input_vector should be a list with len = input neurons
                for i, neuron in enumerate(self.neurons):
                    if neuron.neuron_type == 0:
                        if input_vector[i] > 0:
                            neuron.charge =  int(neuron.charge) + int(input_vector[i]) # add charge to the corresponding neuron
                            self.history.record_incoming(neuron, input_vector[i], "input")
        elif self.spike_train and self.t_step < len(self.spike_train):
            if self.spike_train[self.t_step] == 1: # one boolean spike train for all the input neurons
                input_spike = True
                for neuron in self.neurons:
                    if neuron.neuron_type == 0:
                        neuron.charge += 1
                        self.spike_fired += 1
                        self.history.record_incoming(neuron, 1, "input")

        # consume current spiking events
        for spike_event in self.spike_events[self.t_step % self.max_delay]:
            for idx in spike_event.targets:
                if idx >= 0:
                    self.neurons[idx].receive(spike_event.charge)
                    #self.history.record_incoming(self.neurons[idx], spike_event.charge, spike_event.nid)
                else:
                    self.neurons[-idx].inhibit(spike_event.charge)
                    #self.history.record_incoming(self.neurons[-idx], spike_event.charge, spike_event.nid)
                self.history.record_incoming(self.neurons[idx], spike_event.charge, spike_event.nid)

        # fill charge maps
        if Config.MODE == "CNN":
            if Config.NUM_LAYERS - 3 < self.t_step <= len(self.spike_train) + Config.NUM_LAYERS - 3:
                for input_id in range(Config.NEURONS_L3): #generate pooling images
                    offset = input_id + Config.NEURONS_L1 + Config.NEURONS_L2
                    self.pooling_image[input_id][self.t_step - Config.NUM_LAYERS + 2] = self.neurons[offset].charge
            if Config.NUM_LAYERS - 2 < self.t_step <= len(self.spike_train) + Config.NUM_LAYERS - 2:
                if len(self.labels) == 0: #check if this is the test phase
                    class_id = 0
                    for neuron in self.neurons:
                        if neuron.neuron_type == 2:
                            self.charge_map_prediction[class_id][self.t_step - Config.NUM_LAYERS + 1] = neuron.charge
                            class_id += 1
                            if class_id == Config.CLASSES:
                                break


        # clear current spiking events
        self.spike_events[self.t_step % self.max_delay].clear()

        # check for halting computation
        any_in_delay = any(n.refractory > 0 for n in self.neurons)
        any_spike_in_transit = any(self.spike_events[i] for i in range(self.max_delay))
        if Config.MODE == "generative" and len(self.output) == 2: # halt computation if output has fired 2 times
            return False
        if not any_rule_applied and not any_in_delay and not any_spike_in_transit and not input_spike and self.t_step > 1:
            if Config.MODE == "halting":
                print("The computation halts because no further rules can be applied; the input is accepted")
            return False # end computation

        self.t_step += 1 # advance time

        self.timer.end_step()

        if self.t_step > self.max_steps:
            if Config.MODE == "halting":
                print("The system did not halt naturally within the given step bound; the input is rejected")
            return False # end computation
        else:
            return True # continue computation


    def load_neurons_from_csv(self, filename):
        """Read a CSV file and create the corresponding SNPS"""
        neurons = []

        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                if not any(row):
                    break

                initial_charge = int(row[1])
                targets = eval(row[2])
                neuron_type = int(row[3])
                # Read rules
                transf_rules = []
                for cell in row[4:]:
                    if not cell.strip():
                        break
                    try:
                        values = eval(cell)
                        if len(values) != 5:
                            raise ValueError(f"Wrong rules: {cell}")
                        rule = TransformationRule(values[0],values[1],values[2],values[3],values[4])
                        transf_rules.append(rule)
                    except Exception as e:
                        print(f"Error during rule reading {cell}: {e}")

                # Create neuron
                neuron = PNeuron(snp_system=self, charge = initial_charge, targets=targets, transf_rules=transf_rules, neuron_type=neuron_type)
                neurons.append(neuron)
        self.neurons = neurons
        self.output_neuron_ids = [n.nid for n in neurons if n.neuron_type == 2]

        # Keep prediction matrix aligned with current run length.
        if Config.MODE == "CNN" and self.spike_train is not None:
            n_samples = len(self.spike_train)
            self.charge_map_prediction = np.zeros((Config.CLASSES, n_samples), dtype=int)

        return neurons