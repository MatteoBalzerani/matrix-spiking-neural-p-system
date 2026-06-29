import time
import torch
from datetime import datetime
import csv
from pathlib import Path
from sps.config import Config

class AccuracyLogger:
    """Log accuracy in a CSV file"""
    
    DIR_NAME = "results"
    
    def __init__(self, filename="accuracy_results.csv", overwrite=True):
        self.filename = filename
        self.rows = []
        self.base_dir = Path.cwd()
        self.overwrite = overwrite
        
        if self.overwrite:
            self._load_existing_results()
    
    def _load_existing_results(self):
        """Look for previous results"""
        results_dir = self.base_dir / self.DIR_NAME
        
        if not results_dir.exists():
            return
        
        base_name = self.filename.replace('.csv', '')
        existing_files = list(results_dir.glob(f"{base_name}*.csv"))
        
        if existing_files:
            latest_file = max(existing_files, key=lambda x: x.stat().st_mtime)
            print(f"Loading existing results from: {latest_file}")
            
            try:
                with open(latest_file, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        self.rows.append(row)
                print(f"Loaded {len(self.rows)} existing results")
            except Exception as e:
                print(f"Error loading existing results: {e}")
                self.rows = []
    
    def add_result(self, system, device, q_range, test_num, seed, train_size, test_size, accuracy, num_kernels=None, sparsity=None, positive=None, error=None):
        self.rows.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'system': system,
            'device': device,
            'Q_RANGE': q_range,
            'TEST_NUM': test_num,
            'SEED': seed,
            'TRAIN_SIZE': train_size,
            'TEST_SIZE': test_size,
            'NUM_KERNELS': num_kernels if num_kernels is not None else '',
            'SPARSITY': sparsity if sparsity is not None else '',
            'POSITIVE': positive if positive is not None else '',
            'ACCURACY': f"{accuracy:.4f}" if error is None else 'ERROR',
            'ERROR': str(error)[:200] if error else ''
        })
    
    def save(self):
        """Save each results in CSV file, overwriting the previous fine, if necessary"""
        if not self.rows:
            print("No results to save")
            return None
        
        results_dir = self.base_dir / self.DIR_NAME
        results_dir.mkdir(parents=True, exist_ok=True)
        
        if self.overwrite:
            csv_path = results_dir / self.filename
            print(f"Overwriting existing file: {csv_path}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = results_dir / f"{self.filename.replace('.csv', '')}_{timestamp}.csv"
        
        fieldnames = ['timestamp', 'system', 'device', 'Q_RANGE', 'TEST_NUM', 'SEED', 
              'TRAIN_SIZE', 'TEST_SIZE', 'NUM_KERNELS', 'SPARSITY', 'POSITIVE', 'ACCURACY', 'ERROR']
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        
        print(f"Accuracy results saved to: {csv_path}")
        print(f"Total results: {len(self.rows)}")
        return csv_path
    
    def save_backup(self):
        """Save a backup copy with timestamp (optional)."""
        if not self.rows:
            return None
        
        results_dir = self.base_dir / self.DIR_NAME
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = results_dir / f"{self.filename.replace('.csv', '')}_backup_{timestamp}.csv"
        
        fieldnames = ['timestamp', 'system', 'device', 'Q_RANGE', 'TEST_NUM', 'SEED', 
              'TRAIN_SIZE', 'TEST_SIZE', 'NUM_KERNELS', 'SPARSITY', 'POSITIVE', 'ACCURACY', 'ERROR']
        
        with open(backup_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        
        print(f"Backup saved to: {backup_path}")
        return backup_path


class TimerSNP:
    DIR_NAME = "times"
    
    def __init__(self, expected_steps=100, FILENAME="performance_times.csv", use_cuda=False):
        self._buffer = [None] * expected_steps
        self._index = 0
        self._start_times = [0.0] * expected_steps
        self._step_names = [None] * expected_steps
        self.FILENAME = FILENAME
        self.use_cuda = use_cuda
        
        if use_cuda:
            self._start_events = [None] * expected_steps
            self._end_events = [None] * expected_steps
    
    def start_step(self, step_name):
        self._step_names[self._index] = step_name
        
        if self.use_cuda and torch.cuda.is_available():
            self._start_events[self._index] = torch.cuda.Event(enable_timing=True)
            self._end_events[self._index] = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            self._start_events[self._index].record()
        else:
            self._start_times[self._index] = time.perf_counter()
    
    def end_step(self):
        if self._step_names[self._index] is not None:
            if self.use_cuda and torch.cuda.is_available():
                self._end_events[self._index].record()
                torch.cuda.synchronize()
                elapsed = self._start_events[self._index].elapsed_time(self._end_events[self._index])
                self._buffer[self._index] = elapsed
            else:
                elapsed = time.perf_counter() - self._start_times[self._index]
                self._buffer[self._index] = elapsed * 1000
            self._index += 1
    
    def export_to_csv(self, putInQfolder=False):
        base_dir = Path(__file__).parent.parent 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if putInQfolder:
            csv_path = base_dir / self.DIR_NAME / f"Q_{Config.Q_RANGE}" / f"{self.FILENAME}_{timestamp}.csv"
        else:
            csv_path = base_dir / self.DIR_NAME / f"{self.FILENAME}_{timestamp}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'Time (ms)']) 
            
            for i in range(self._index):
                if self._buffer[i] is not None:
                    writer.writerow([self._step_names[i], f"{self._buffer[i]:.3f}"])
    
    def export_training_times(self, system_name):
        """
        Esporta i tempi di training in un CSV.
        Le colonne sono per SPARSITY, POSITIVE e TEST_NUM.
        """
        base_dir = Path(__file__).parent.parent
        csv_path = base_dir / self.DIR_NAME / f"training_times_{system_name}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        training_data = {}
        
        for i in range(self._index):
            if self._buffer[i] is not None:
                step_name = str(self._step_names[i])
                time_ms = self._buffer[i]
                
                if 'TRAINING' in step_name:
                    try:
                        # Formato: "SP:{sparsity}_P:{positive}_T:{test_num}_{system}_TRAINING_SVM"
                        parts = step_name.split('_')
                        sparsity = float(parts[0].split(':')[1])
                        positive = float(parts[1].split(':')[1])
                        test_num = int(parts[2].split(':')[1])
                        model_type = parts[-1]  # SVM o LogReg
                        
                        key = (sparsity, positive, test_num)
                        if key not in training_data:
                            training_data[key] = {}
                        training_data[key][model_type] = time_ms
                    except:
                        continue
        
        if not training_data:
            print("No training data to export")
            return
        
        # Carica dati esistenti
        existing_data = {}
        existing_columns = []
        
        if csv_path.exists():
            try:
                with open(csv_path, 'r', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    existing_columns = [c for c in reader.fieldnames[2:] if c]
                    for row in reader:
                        key = (row['Config'], row['Model'])
                        existing_data[key] = {}
                        for col in existing_columns:
                            if row.get(col):
                                try:
                                    existing_data[key][col] = float(row[col])
                                except ValueError:
                                    pass
            except Exception as e:
                print(f"Error loading existing training times: {e}")
                existing_data = {}
                existing_columns = []
        
        # Unisci nuovi dati
        for (sparsity, positive, test_num) in training_data:
            col_name = f"SP{sparsity}_P{positive}_T{test_num}"
            if col_name not in existing_columns:
                existing_columns.append(col_name)
            for model in ['SVM', 'LogReg']:
                time_value = training_data[(sparsity, positive, test_num)].get(model)
                if time_value is not None:
                    config_label = f"SP{sparsity}_P{positive}"
                    key = (config_label, model)
                    if key not in existing_data:
                        existing_data[key] = {}
                    existing_data[key][col_name] = time_value
        
        # Ordina le chiavi
        all_keys = sorted(existing_data.keys(), key=lambda x: (x[0], x[1]))
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            header = ['Config', 'Model'] + existing_columns
            writer.writerow(header)
            
            for key in all_keys:
                row = list(key)
                for col in existing_columns:
                    time_value = existing_data[key].get(col, '')
                    row.append(f"{time_value:.3f}" if time_value != '' else '')
                writer.writerow(row)
        
        print(f"Training times saved to: {csv_path}")
        return csv_path
        
    def export_step_times(self, system_name, timer_type, phase="TRAIN"):
        base_dir = Path(__file__).parent.parent
        sparsity = Config.M_SPARSITY
        positive = Config.M_POSITIVE
        test_num = Config.TIME_TEST_NUM
        size_info = f"{Config.TRAIN_SIZE}-{Config.TEST_SIZE}"
        
        if timer_type == "InStep":
            filename = f"times_{phase}_InStep_{system_name}_S{size_info}.csv"
        else:
            filename = f"times_{phase}_PerStep_{system_name}_S{size_info}.csv"
        
        csv_path = base_dir / self.DIR_NAME / filename
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        step_data = {}
        # Colonna: SP{sparsity}_P{positive}_T{test_num}
        column_key = f"SP{sparsity}_P{positive}_T{test_num}"
        for i in range(self._index):
            if self._buffer[i] is not None:
                step_name = str(self._step_names[i])
                time_ms = self._buffer[i]
                if step_name not in step_data:
                    step_data[step_name] = {}
                step_data[step_name][column_key] = time_ms
        
        # Carica colonne esistenti
        existing_columns = []
        existing_data = {}
        if csv_path.exists():
            try:
                with open(csv_path, 'r', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    existing_columns = [c for c in reader.fieldnames[1:] if c]
                    for row in reader:
                        step = row['Step']
                        existing_data[step] = {}
                        for col in existing_columns:
                            if col and row[col]:
                                try:
                                    existing_data[step][col] = float(row[col])
                                except ValueError:
                                    pass
            except Exception as e:
                print(f"Error loading existing step times: {e}")
                existing_data = {}
                existing_columns = []
        
        for step, times in existing_data.items():
            if step not in step_data:
                step_data[step] = {}
            step_data[step].update(times)
        
        if column_key not in existing_columns:
            existing_columns.append(column_key)
        
        def step_sort_key(step_name):
            try:
                step_name_str = str(step_name)
                if '>' in step_name_str:
                    step_num = int(step_name_str.split('>')[0])
                    if timer_type == "InStep":
                        sub_step = step_name_str.split('>')[1].strip()
                        sub_step_order = {
                            'Image Input': 0,
                            'Extended Config + Spiking Vector construction': 1,
                            'NetGain Vector update: smpi @ spikingVec': 2,
                            'Configuration Vector update': 3,
                            'Pooling image update': 4
                        }
                        sub_order = sub_step_order.get(sub_step, 99)
                        return (step_num, sub_order)
                    else:
                        return (step_num, 0)
                else:
                    return (int(step_name_str), 0)
            except:
                return (999999, 0)
        
        sorted_steps = sorted(step_data.keys(), key=step_sort_key)
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Step'] + existing_columns
            writer.writerow(header)
            for step in sorted_steps:
                row = [step]
                for col in existing_columns:
                    time_value = step_data[step].get(col, '')
                    row.append(f"{time_value:.3f}" if time_value != '' else '')
                writer.writerow(row)
        
        print(f"Step times ({timer_type}) saved to: {csv_path}")
        return csv_path