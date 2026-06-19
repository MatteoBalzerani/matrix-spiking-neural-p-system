import numpy as np
import torch
from sps.config import Config
from sps.system_measurers import TimerSNP

class MSNPSystemExactGPU:

    def __init__(self, configurationVector, spikingVector, sMpi_sparse,
                 ruleVector, max_steps=1000, deterministic=True,
                 single_spike_train=None, input_neurons=None, output_neurons=None,
                 applyingRuleVector=None, device='cpu', testsize=1,debugMode=""):

        if device == 'gpu' or device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.dtype = torch.float32
                print(f"MSNPSystemGPU using device: CUDA GPU with dtype={self.dtype}")
            else:
                print("Warning: CUDA not available. Falling back to CPU.")
                self.device = torch.device('cpu')
                self.dtype = torch.int32
                print(f"MSNPSystemGPU using device: CPU with dtype={self.dtype}")
        else:
            self.device = torch.device('cpu')
            self.dtype = torch.int32
            print(f"MSNPSystemGPU using device: CPU with dtype={self.dtype}")

        self._needs_float_conversion = (self.device.type == 'cpu' and self.dtype == torch.int32)

        if applyingRuleVector is None or configurationVector is None or ruleVector is None:
            raise ValueError("ApplyingRuleVector, configurationVector and ruleVector cannot be None")

        if max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        #rule_num = len(spikingTransitionMatrix)
        rule_num = sMpi_sparse.shape[0]
        neuron_num = len(configurationVector)

        self.testsize = testsize
        self.pooling_image = torch.zeros((len(output_neurons), self.testsize), dtype=self.dtype, device=self.device) if output_neurons is not None else None
        
        if self.pooling_image is not None:
            self._pooling_offset = 1 if len(output_neurons) == Config.CLASSES else 0
            self._pooling_lower_bound = Config.NUM_LAYERS - 4  # bound inferiore originale
            self._pooling_upper_bound = self.testsize + Config.NUM_LAYERS - 4  # bound superiore originale
            self._output_neurons_tensor = torch.tensor(output_neurons, device=self.device)
            self._is_classification = len(output_neurons) == Config.CLASSES
        else:
            self._pooling_offset = 0
            self._pooling_start = 0
            self._pooling_end = 0
            self._output_neurons_tensor = None
            self._is_classification = False 

        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.configurationVector = torch.tensor(configurationVector, dtype=self.dtype, device=self.device)
        self.netGainVector = torch.zeros(neuron_num, dtype=self.dtype, device=self.device)
        self.ruleVector = torch.tensor(ruleVector, dtype=self.dtype, device=self.device)
        self.applyingRuleVector = torch.tensor(applyingRuleVector, dtype=torch.int32, device=self.device)
        self.sMpi = sMpi_sparse.to(self.device)

        if self._needs_float_conversion:
            self._sMpi_float = self.sMpi.float().to_sparse_csr()
        else:
            if self.sMpi.layout == torch.sparse_coo:
                self.sMpi = self.sMpi.to_sparse_csr()
            self._sMpi_float = None

        self.ruleCountPerNeuron = torch.bincount(self.applyingRuleVector, minlength=neuron_num)
        self.neuron_idx_expanded = torch.repeat_interleave(
            torch.arange(neuron_num, device=self.device),
            self.ruleCountPerNeuron
        )

        if spikingVector is None:
            self.spikingVector = torch.zeros(rule_num, dtype=self.dtype, device=self.device)
        else:
            self.spikingVector = torch.tensor(spikingVector, dtype=self.dtype, device=self.device)

        if single_spike_train is not None:
            self.single_spike_train = torch.tensor(single_spike_train, dtype=self.dtype, device=self.device)
        else:
            self.single_spike_train = torch.tensor([], dtype=self.dtype, device=self.device)

        ##################################
        # TIMER SETUP 
        #
        # Define name and phase
        if self.device == torch.device('cuda'):
            system_name = "MSNPSystem_GPU"
            use_cuda_timer = True
        else:
            system_name = "MSNPSystem_CPU"
            use_cuda_timer = False
        
        # debugMode will contain "TRAIN" or "TEST"
        self._system_name = system_name
        self._phase = debugMode  # "TRAIN" or "TEST"
        
        self.timerInStep = TimerSNP(
            self.max_steps * 10,
            f"{debugMode}_{Config.TRAIN_SIZE}-{Config.TEST_SIZE}_T{Config.TIME_TEST_NUM}_time_InStep_{system_name}",
            use_cuda_timer
        )
        self.timerPerStep = TimerSNP(
            self.max_steps,
            f"{debugMode}_{Config.TRAIN_SIZE}-{Config.TEST_SIZE}_T{Config.TIME_TEST_NUM}_time_PerStep_{system_name}",
            use_cuda_timer
        )

        ###################

        self.t_step = 0
      
 
    def loadImages(self, img_spike_train):
        """Load images as spike trains for CNN mode"""
        if len(img_spike_train.shape) == 3:
            img_spike_train = img_spike_train.reshape(img_spike_train.shape[0], -1)
        self.img_spike_train = torch.tensor(img_spike_train, dtype=self.dtype, device=self.device)

    def step(self, verbose=False):
        """Execute one step of the system"""
        
        self.timerInStep.start_step(f"{self.t_step}> Image Input")
        self.timerPerStep.start_step(self.t_step)

        # 1. Input spikes or image in the system -----------------------------------
        if Config.MODE == "CNN":
            if self.t_step < self.img_spike_train.shape[0]:
                self.configurationVector[self.input_neurons] += self.img_spike_train[self.t_step]
        elif self.single_spike_train.size(0) > 0 and self.t_step < self.single_spike_train.shape[0]:
            spike_value = 1 if self.dtype == torch.int32 else 1.0
            if self.single_spike_train[self.t_step] == spike_value:
                self.configurationVector[self.input_neurons] += spike_value
        self.timerInStep.end_step()

        # 2-3 Calculate extended configuration vector + Spiking vector ------------------------
        self.timerInStep.start_step(f"{self.t_step}> Extended Config + Spiking Vector construction")
        #  Extended config + Spiking vector 
        diff = torch.abs(self.configurationVector[self.neuron_idx_expanded] - self.ruleVector)
        self.spikingVector = torch.div(1, 1 + diff, rounding_mode='floor') if self.dtype == torch.int32 \
            else torch.floor(1.0 / (1.0 + diff))
        self.timerInStep.end_step()
        

        # 4. Update netgain vector vector - MATRIX MULTIPLICATION -----------------
        self.timerInStep.start_step(f"{self.t_step}> NetGain Vector update: smpi @ spikingVec")
        if self._needs_float_conversion:
            self.netGainVector = (self.spikingVector.float() @ self._sMpi_float).to(torch.int32)
        else:
            self.netGainVector = self.spikingVector @ self.sMpi
        self.timerInStep.end_step()

        # 5. Configuration Vector update ------------------------------------------
        self.timerInStep.start_step(f"{self.t_step}>Configuration Vector update")
        self.configurationVector += self.netGainVector
        self.timerInStep.end_step()

        # 6. Pooling image update for training
        self.timerInStep.start_step(f"{self.t_step}>Pooling image update")
        
        if self.pooling_image is not None:
            t_effective = self.t_step - self._pooling_offset
            
            if self._pooling_lower_bound < t_effective <= self._pooling_upper_bound:
                col = t_effective - Config.NUM_LAYERS + 3  # Formula originale invariata
                self.pooling_image[:, col] = self.configurationVector[self._output_neurons_tensor]
                
                if self._is_classification:
                    self.configurationVector[self._output_neurons_tensor] = 0
        self.timerInStep.end_step()

        self.timerPerStep.end_step()
        self.t_step += 1
        return True
    
    # old method for updating pooling image
    # if self.pooling_image is not None:
    #     # Nel test (10 classi) la propagazione richiede 1 step in più
    #     offset = 1 if len(self.output_neurons) == Config.CLASSES else 0

    #     if Config.NUM_LAYERS - 4 < self.t_step - offset <= self.testsize + Config.NUM_LAYERS - 4:
    #         col = (self.t_step - offset) - Config.NUM_LAYERS + 3
    #         self.pooling_image[:, col] = self.configurationVector[self.output_neurons]
    #         if len(self.output_neurons) == Config.CLASSES:
    #             self.configurationVector[self.output_neurons] = 0

    def execute(self, verbose=False, startAgain=True):
        if startAgain:
            self.t_step = 0

        if Config.MODE == "CNN":
            input_length = self.img_spike_train.shape[0]
        else:
            input_length = len(self.single_spike_train) if hasattr(self.single_spike_train, '__len__') else 0

        while self.step(verbose=verbose) and (self.t_step < self.max_steps or self.t_step < input_length):
            if torch.all(self.spikingVector == 0) and (self.t_step >= input_length):
                print("Computation halts: spiking vector is zero, input is accepted")
                np.save("/tmp/charge_map_gpu.npy", self.pooling_image.cpu().numpy())
                print(f"Saved charge_map_gpu: {self.pooling_image.shape}")
                
                
                # Export times in CSV file
                self.timerInStep.export_step_times(self._system_name, "InStep", self._phase)
                self.timerPerStep.export_step_times(self._system_name, "PerStep", self._phase)
                
                return True

        print("Computation halts: maximum number of steps reached, input is rejected")
        return False

    def get_pooling_image(self):
        """Returns the pooling image as a numpy array"""
        if self.pooling_image is not None:
            return self.pooling_image.cpu().numpy()
        return None

    def get_configuration_vector(self):
        return self.configurationVector.cpu().numpy()

    def get_spiking_vector(self):
        return self.spikingVector.cpu().numpy()

    def get_net_gain_vector(self):
        return self.netGainVector.cpu().numpy()

    def get_spiking_transition_matrix(self):
        return self.sMpi.cpu().numpy()

    def get_rule_vector(self):
        return self.ruleVector.cpu().numpy()

    def get_pooling_image_tensor(self):
        """Returns the pooling image tensor on the current device (no CPU transfer)"""
        return self.pooling_image

    def get_applying_rule_vector(self):
        return self.applyingRuleVector.cpu().numpy()

    def __str__(self):
        return (f"Device: {self.device} (dtype: {self.dtype})\n"
                f"Deterministic: {self.deterministic}\n"
                f"Input Neurons: {self.input_neurons}\n"
                f"Output Neurons: {self.output_neurons}\n"
                f"Configuration Vector: {self.get_configuration_vector()}\n"
                f"Spiking Vector: {self.get_spiking_vector()}\n"
                f"Net Gain Vector: {self.get_net_gain_vector()}\n"
                f"Rule Vector: {self.get_rule_vector()}\n"
                f"Applying Rule Vector: {self.get_applying_rule_vector()}\n")
