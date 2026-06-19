import csv
import pandas as pd
import os

# mode can be "step_by_step" or "all_at_once"
# format can be "csv" or "parquet"

class ChargeTracker:
    def __init__(self, filename, mode="all_at_once", format="parquet", num_neurons=0):
        self.filename = f"{filename}.{format}"
        self.mode = mode
        self.format = format
        self.num_neurons = num_neurons
        self.history = [] # used only for all_at_once mode
        self.csv_file = None
        self.csv_writer = None
        
        self.header = ["Image"] + [str(i) for i in range(num_neurons)] # neurons' numbers are in the first row of the file, image numbers are in the first column

        # initializes the file if we are in step-by-step mode
        if self.mode == "step_by_step" and self.format == "csv":
            self.csv_file = open(self.filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            header = ["Step"] + [f"N_{i}" for i in range(num_neurons)]
            self.csv_writer.writerow(header)
            
        elif self.mode == "step_by_step" and self.format == "parquet":
            print("the step-by-step saving in Parquet is slow.") #it would require libraries like fastparquet with append=True, better to use CSV for step-by-step.

    def record_charges(self, image_index, neurons):
        row = [image_index] + [n.charge for n in neurons]
        
        if self.mode == "step_by_step":
            if self.format == "csv":
                self.csv_writer.writerow(row)
            elif self.format == "parquet":
                # Buffer rows and flush once in finish().
                self.history.append(row)
        else:
            self.history.append(row)

    def finish(self): # closes the file or writes the entire file (all_at_once)
        if self.mode == "step_by_step":
            if self.format == "csv" and self.csv_file is not None:
                self.csv_file.close()
            elif self.format == "parquet":
                df = pd.DataFrame(self.history, columns=self.header)
                try:
                    df.to_parquet(self.filename, engine='pyarrow', index=False)
                except (ImportError, ModuleNotFoundError):
                    fallback_filename = os.path.splitext(self.filename)[0] + ".csv"
                    print("[ChargeTracker] pyarrow missing, saving CSV:", fallback_filename)
                    df.to_csv(fallback_filename, index=False)
                self.history.clear()
                
        elif self.mode == "all_at_once":
            df = pd.DataFrame(self.history, columns=self.header)
            
            if self.format == "csv":
                df.to_csv(self.filename, index=False)
            elif self.format == "parquet":
                try:
                    df.to_parquet(self.filename, engine='pyarrow', index=False)
                except (ImportError, ModuleNotFoundError):
                    fallback_filename = os.path.splitext(self.filename)[0] + ".csv"
                    print("[ChargeTracker] pyarrow missing, saving CSV:", fallback_filename)
                    df.to_csv(fallback_filename, index=False)
            
            self.history.clear() # clears memory after writing to disk