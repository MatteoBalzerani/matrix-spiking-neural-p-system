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


def emergency_save(generalTimer, accuracyLogger, error_msg=""):
    """Emergency save of results"""
    try:
        generalTimer.export_to_csv(False)
        accuracyLogger.save()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = f"error_log_{timestamp}.txt"
        with open(error_file, 'w') as f:
            f.write(f"Error: {error_msg}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
        
        print(f"\n{'!'*50}")
        print(f"EMERGENCY SAVE completed!")
        print(f"Timer and Accuracy saved")
        print(f"Error log: {error_file}")
        print(f"{'!'*50}")
        
    except Exception as e:
        print(f"EMERGENCY SAVE FAILED: {e}")


def run_system_test(system_name, device, Q_RANGE, TEST_NUM, SEED, size, generalTimer, accuracyLogger):
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
        generalTimer.start_step(step_name)
        accuracy = network.launch_mnist(system_name, device)
        generalTimer.end_step()
        
        accuracyLogger.add_result(system_label, device.upper(), Q_RANGE, TEST_NUM, SEED, 
                                 size[0], size[1], accuracy)
        print(f"<<< DONE [{system_label} - {device.upper()}] Accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        error_msg = f"{system_label} {device.upper()} - Q:{Q_RANGE} T:{TEST_NUM} Size:{size}"
        print(f"\n{'!'*50}")
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc())
        print(f"{'!'*50}")
        accuracyLogger.add_result(system_label, device.upper(), Q_RANGE, TEST_NUM, SEED,
                                 size[0], size[1], None, error=error_msg)
        emergency_save(generalTimer, accuracyLogger, error_msg)
        return False


# ============================================================
# MAIN
# ============================================================

try:
    reset_gpu_for_rerun()
    
    database("digit")
    Config.compute_k_range()
    
    sizes = [[50,50],[1000,100],[5000,2500]]
    Q_LIMIT = 8
    TEST_PER_Q = 3
    SEEDS = [42, 999, 1234]
    
    total_tests = Q_LIMIT * TEST_PER_Q * len(sizes) * 3  # 3 systems
    generalTimer = sps.system_measurers.TimerSNP(total_tests, "ElapsedTimePerSystemAndQrange", True)
    accuracyLogger = sps.system_measurers.AccuracyLogger("accuracy_results.csv", True)
    
    test_counter = 0
    
    # ========================================================
    # PHASE 1: ALL SNPS WITH CPU
    # ========================================================
    print("\n" + "="*60)
    print("PHASE 1: EXECUTING ALL SNPS WITH CPU")
    print("="*60)

    for size in sizes:
        for Q_RANGE in range(2, Q_LIMIT + 1):
            for TEST_NUM in range(0, TEST_PER_Q):
            
                reset_gpu_for_rerun()
                SEED = SEEDS[TEST_NUM]
                
                run_system_test("SNPSystem", "cpu", Q_RANGE, TEST_NUM, SEED, size, 
                               generalTimer, accuracyLogger)
                
                test_counter += 1
                if test_counter % 9 == 0:
                    accuracyLogger.save()
                    print(f"Partial results saved! ({test_counter} tests completed)")
    
    # ========================================================
    # PHASE 2: ALL MSNPS WITH GPU
    # ========================================================
    print("\n" + "="*60)
    print("PHASE 2: EXECUTING ALL MSNPS WITH GPU")
    print("="*60)
    
    for size in sizes:
        for Q_RANGE in range(2, Q_LIMIT + 1):
            for TEST_NUM in range(0, TEST_PER_Q):
                reset_gpu_for_rerun()
                kill_all_cuda_contexts()
                SEED = SEEDS[TEST_NUM]
                
                run_system_test("MSNPSystemExactGPU", "gpu", Q_RANGE, TEST_NUM, SEED, size, 
                               generalTimer, accuracyLogger)
                
                test_counter += 1
                if test_counter % 9 == 0:
                    accuracyLogger.save()
                    print(f"Partial results saved! ({test_counter} tests completed)")
    
    # ========================================================
    # PHASE 3: ALL MSNPS WITH --CPU--
    # ========================================================
    print("\n" + "="*60)
    print("PHASE 3: EXECUTING ALL MSNPS WITH CPU")
    print("="*60)
    
    for size in sizes:
        for Q_RANGE in range(2, Q_LIMIT + 1):
            for TEST_NUM in range(0, TEST_PER_Q):
                reset_gpu_for_rerun()
                kill_all_cuda_contexts()
                SEED = SEEDS[TEST_NUM]
                
                run_system_test("MSNPSystemExactGPU", "cpu", Q_RANGE, TEST_NUM, SEED, size, 
                               generalTimer, accuracyLogger)
                
                test_counter += 1
                if test_counter % 9 == 0:
                    accuracyLogger.save()
                    print(f"Partial results saved! ({test_counter} tests completed)")

except KeyboardInterrupt:
    print("\n\nINTERRUPTED BY USER!")
    error_msg = "KeyboardInterrupt"
    emergency_save(generalTimer, accuracyLogger, error_msg)
    
except Exception as e:
    print("\n\nFATAL ERROR!")
    error_msg = f"FATAL: {str(e)}"
    print(traceback.format_exc())
    emergency_save(generalTimer, accuracyLogger, error_msg)

finally:
    # Salva sempre alla fine
    print("\n\n" + "="*50)
    print("SAVING FINAL RESULTS...")
    try:
        generalTimer.export_to_csv(False)
        accuracyLogger.save()
        accuracyLogger.save_backup()  # Backup finale
        print("Final results saved successfully!")
    except Exception as e:
        print(f"Error saving final results: {e}")
    
    print("="*50)
    print("EXECUTION COMPLETED")