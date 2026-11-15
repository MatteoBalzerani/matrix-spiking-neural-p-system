import os
import sys
import traceback
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datetime import datetime

from sps import  network
from sps.config import Config, database
import torch
import gc
import sps.system_measurers
import random

def reset_gpu_for_rerun():
    """Prepare GPU for rerun"""
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    obj.data = obj.data.cpu()
                    del obj
        except:
            pass
    
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith('sps.'):
            for attr_name in dir(mod):
                try:
                    attr = getattr(mod, attr_name)
                    if hasattr(attr, '_buffers'):
                        for buf in attr._buffers.values():
                            if torch.is_tensor(buf) and buf.is_cuda:
                                buf.data = buf.data.cpu()
                except:
                    pass
    
    gc.collect()
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory after reset - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def kill_all_cuda_contexts():
    """Start again CUDA environment"""
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        for i in range(device_count):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        print("CUDA contexts reset completely")
        
def run_system_test(system_name, device, Q_RANGE, TEST_NUM, SEED, size, generalTimer = None, accuracyLogger = None):
    """ Launch the specified system with given parameters"""
    Config.TRAIN_SIZE = size[0]
    Config.TEST_SIZE = size[1]
    
    print("\n" + "#"*50)
    print(f"SYSTEM: {system_name} | DEVICE: {device}")
    print(f"TRAIN SIZE: {Config.TRAIN_SIZE} | TEST SIZE: {Config.TEST_SIZE}")
    print(f"Q: {Q_RANGE} | TEST: {TEST_NUM} | SEED: {SEED}")
    print("#"*50)
    
    system_label = "SNPS" if "SNPSystem" in system_name else "MSNPS"
    
    try:
        reset_gpu_for_rerun()
        kill_all_cuda_contexts()
        
        # Set seed
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"\n>>> RUNNING [{system_label} - {device.upper()}] Q:{Q_RANGE} T:{TEST_NUM}")
        Config.TIME_TEST_NUM = TEST_NUM
        Config.Q_RANGE = Q_RANGE

        step_name = f"Q:{Q_RANGE}_T:{TEST_NUM}_S{size[0]}_{system_label}_{device.upper()}"
        if generalTimer:
            generalTimer.start_step(step_name)

        accuracy = network.launch_mnist(system_name, device)

        if generalTimer:
            generalTimer.end_step()

        if accuracyLogger:
            accuracyLogger.add_result(system_label, device.upper(), Q_RANGE, TEST_NUM, SEED,
                                size[0], size[1], accuracy,
                                sparsity=Config.M_SPARSITY, positive=Config.M_POSITIVE)
        print(f"<<< DONE [{system_label} - {device.upper()}] Accuracy: {accuracy:.4f}")

    except Exception as e:
        print(f"\n\nFATAL ERROR in [{system_label} - {device.upper()}]: {str(e)}")

# ============================================================
# MAIN
#
# Example of a phantom execution of the system, 
# to test the different configurations.
# ============================================================

try:
    reset_gpu_for_rerun()
    kill_all_cuda_contexts()
        
    database("digit")
  
    Config.TERNARIZE_METHOD = 1
    # TRAIN - TEST
    size = [100, 100]
    Q_RANGE = 3
    SEED = 123
    TEST_NUM = 1

    Config.compute_k_range()

    run_system_test("MSNPSystemExactGPU", "gpu", Q_RANGE, TEST_NUM, SEED, size)
    
    run_system_test("SNPSystem", "cpu", Q_RANGE, TEST_NUM, SEED, size)

    run_system_test("MSNPSystemExactGPU", "cpu", Q_RANGE, TEST_NUM, SEED, size)
      
except KeyboardInterrupt:
    print("\n\nINTERRUPTED BY USER!")
    error_msg = "KeyboardInterrupt"
    
except Exception as e:
    print("\n\nFATAL ERROR!")
    error_msg = f"FATAL: {str(e)}"