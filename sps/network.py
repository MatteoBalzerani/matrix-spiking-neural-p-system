import os
import numpy as np
import time
import torch
from sklearn.linear_model import LogisticRegression
from sps import handle_csv, exact_csv
from sps.digit_image import get_mnist_data
from sps.exact_csv import SNPS_exact_csv
from sps.handle_csv import SNPS_csv, extend_csv, ensemble_csv
from sps.config import Config
from sps.m_snp_pytorch_exact_GPU_and_CPU import MSNPSystemExactGPU
from sps.snp_system import SNPSystem
from sklearn.svm import LinearSVC
from sps.m_matrix_executor_exact import MatrixExecutor as MatrixExecutorExact
from sps.classifiers_gpu import LogisticRegressionGPU, SVMGPU
from sps.system_measurers import TimerSNP

#temporary code for create the csv and the SNPS with exact rules for the GPU
def create_exact_csv():
    x_train, y_train, x_test, y_test = get_mnist_data()
    SNPS_csv() #create the csv for the SNPS
    svm, logreg = train_SNPS(x_train, y_train)

    snps = SNPSystem(Config.TEST_SIZE, Config.TEST_SIZE + 5, True)
    snps.spike_train = x_test
    svm_q = ternarize_matrix(svm.coef_.T)
    logreg_q = ternarize_matrix(logreg.coef_.T)
    extended_path = exact_csv.ensemble_exact_csv(np.array(svm_q), np.array(logreg_q), get_importance(svm.coef_), get_importance(logreg.coef_))
    snps.load_neurons_from_csv(extended_path)

    return snps



def launch_mnist_from_csv(csv_name):
    x_train, y_train, x_test, y_test = get_mnist_data()

    snps = SNPSystem(Config.TEST_SIZE, Config.TEST_SIZE + 5, True)
    snps.spike_train = x_test

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # folder where this .py file lives
    csv_path = os.path.join(project_root, "csv", csv_name)
    snps.load_neurons_from_csv(csv_path)
    snps.start()
    y_pred = np.argmax(snps.charge_map_prediction, axis=0)

    cnn_accuracy = np.mean(y_pred == y_test)
    print("SNPS with csv: ", csv_name, " get accuracy of:", cnn_accuracy)

def launch_mnist(system, device):
    x_train, y_train, x_test, y_test = get_mnist_data()
    t=time.time()
    #example_direct(x_train, y_train, x_test, y_test) #compare with models baseline, launched directly on input images

    SNPS_csv() #create the csv for the SNPS
    svm, logreg = train_SNPS(system, device, x_train, y_train)
    print("-> Done the training procedure")

    ensemble_accuracy = test_SNPS(system, device, x_test, y_test, svm, logreg)
    #handle_csv.save_results(ensemble_accuracy, time.time()-t)
    return ensemble_accuracy





def train_SNPS(system, device, x_train, y_train):

    snps = SNPSystem(Config.TRAIN_SIZE, Config.TRAIN_SIZE + 5, True,"TRAIN")


    if system == "MSNPSystemExactGPU":
        SNPS_exact_csv()
        snps.load_neurons_from_csv("csv/" + Config.CSV_EXACT_NAME)
        print("csv loading complete")

        # As second parameter, please pass to translate_to_matrix the used device, cpu or gpu
        msnpsExact = MatrixExecutorExact.translate_to_matrix(snps, device,"TRAIN")
        print("translation to matrix complete")
        msnpsExact.loadImages(x_train)
        msnpsExact.execute()

        pooling = msnpsExact.get_pooling_image().T
        print("pooling_image dtype:", pooling.dtype)
        print("pooling_image min/max:", pooling.min(), pooling.max())
        print("pooling_image mean:", pooling.mean())
        print("non-zero count:", np.count_nonzero(pooling))
        np.save("/tmp/pooling_gpu.npy", pooling)
        print("saved to /tmp/pooling_gpu.npy")

        return train_external_models(pooling, y_train, 'cpu')


    #snps.labels = y_train
    elif system == "SNPSystem":


        SNPS_exact_csv()
        snps.load_neurons_from_csv("csv/" + Config.CSV_EXACT_NAME) #rules in exact form
        #snps.load_neurons_from_csv("csv/" + Config.CSV_NAME)

        snps.spike_train = x_train
        snps.start()
        pooling = snps.pooling_image.T

        print("=== SNPSystem DEBUG ===")
        print("pooling_image dtype:", pooling.dtype)
        print("pooling_image min/max:", pooling.min(), pooling.max())
        print("pooling_image mean:", pooling.mean())
        print("non-zero count:", np.count_nonzero(pooling))
        np.save("/tmp/pooling_snp.npy", pooling)
        print("saved to /tmp/pooling_snp.npy")
        return train_external_models(pooling, y_train, device,system)



def train_external_models(charges, y_train, device='cpu', system=""):
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    
    timerTraining = TimerSNP(10, f"T{Config.TIME_TEST_NUM}_TRAINING_time_{system}", False)
    
    # Support Vector Machine
    timerTraining.start_step(f"Q:{Config.Q_RANGE}_T:{Config.TIME_TEST_NUM}_S{Config.TRAIN_SIZE}_{system}_TRAINING_SVM")
    svm = LinearSVC(C=Config.SVM_C, max_iter=10000)
    svm.fit(charges, y_train)
    print("SVM done")
    timerTraining.end_step()
    
    # Logistic Regression
    timerTraining.start_step(f"Q:{Config.Q_RANGE}_T:{Config.TIME_TEST_NUM}_S{Config.TRAIN_SIZE}_{system}_TRAINING_LogReg")
    logreg = LogisticRegression(
        solver="lbfgs",
        max_iter=100000
    )
    logreg.fit(charges, y_train)
    print("LogReg done")
    timerTraining.end_step()
    
    # Export training times
    timerTraining.export_training_times(system)
    
    return svm, logreg


def test_SNPS(system, device, x_test, y_test, svm, logreg):

    #compare_performance(x_test, y_test, svm, logreg) #for running and checking performance of all the other networks

    ensemble_pred = ensemble_and_test(system, device, x_test, svm.coef_, logreg.coef_, get_importance(svm.coef_), get_importance(logreg.coef_))
    ensemble_accuracy = np.mean(ensemble_pred == y_test)
    print("SNPS ensemble accuracy with importance:", ensemble_accuracy)

    return ensemble_accuracy

def ensemble_and_test(system, device, x_test, svm_w, logreg_w, svm_imp, logreg_imp):

    snps = SNPSystem(Config.TEST_SIZE, Config.TEST_SIZE + 5, True,"TEST")
    svm_q = ternarize_matrix(svm_w.T)
    logreg_q = ternarize_matrix(logreg_w.T)

    if system == "MSNPSystemExactGPU":

        extended_path = exact_csv.ensemble_exact_csv(np.array(svm_q), np.array(logreg_q), svm_imp, logreg_imp)
        snps.load_neurons_from_csv(extended_path)

        msnpsExact = MatrixExecutorExact.translate_to_matrix(snps, device,"TEST")
        msnpsExact.loadImages(x_test)
        msnpsExact.execute()

        charge_map = msnpsExact.pooling_image.cpu().numpy()  # shape (10, testsize)

        y_pred = np.argmax(charge_map, axis=0)
        return y_pred

    elif system == "SNPSystem":
        snps.spike_train = x_test

        extended_path = exact_csv.ensemble_exact_csv(np.array(svm_q), np.array(logreg_q), svm_imp, logreg_imp)  #rules in exact form
        #extended_path = ensemble_csv(np.array(svm_q), np.array(logreg_q), svm_imp, logreg_imp)

        snps.load_neurons_from_csv(extended_path)

        snps.start()
        y_pred = np.argmax(snps.charge_map_prediction, axis=0)
        return y_pred



def get_importance(w):
    imp = compute_neuron_importance(w)
    if Config.DISCRETIZE_METHOD == 1: multipliers = discretize_percentile(imp)
    else: multipliers = discretize_proportional(imp) #Config.DISCRETIZE == 2
    return multipliers

def compute_neuron_importance(w):
    #Magnitude: neurons with larger overall weights across classes are considered more influential
    if Config.IMPORTANCE_METHOD == 1: imp = np.linalg.norm(w, axis=0)
    # Weight range: neurons whose weights vary more between classes are considered more discriminative
    else: imp = np.max(w, axis=0) - np.min(w, axis=0) #Config.IMPORTANCE_METHOD == 2
    imp = imp / max(imp.max(), 1e-8) #Normalize in range [0:1]
    return imp

def discretize_percentile(imp): #method 1 - percentile based importance
    p25 = np.percentile(imp, 25)   # first quartile
    p75 = np.percentile(imp, 75)   # third quartile
    multipliers = np.ones_like(imp)  # default multiplier = 1 (low importance)
    multipliers[imp > p75] = 3       # top 25% most important neurons
    multipliers[(imp > p25) & (imp <= p75)] = 2  # middle 50%
    multipliers[imp <= p25] = 1      # bottom 25%
    return multipliers.astype(int) # convert to integer

def discretize_proportional(imp): #method 2 - proportional based importance
    multipliers = 1 + np.round(imp * Config.DISC_RANGE)
    return multipliers.astype(int) # convert to integer



def ternarize_matrix(w):
    # matrix quantization for last layer of SNPS: Transform from real values to {-1,0,1}
    if Config.TERNARIZE_METHOD == 1: q = ternarize_percentile(w, Config.M_SPARSITY, Config.M_POSITIVE) # Percentile-based
    else: q = ternarize_threshold(w, Config.M_THRESHOLD) # Threshold-based

    np.save("ternary_matrix.npy", q)

    return q

#for more info see https://www.emergentmind.com/topics/ternary-weight-networks-twns
def ternarize_percentile(w, p_zero, p_pos):
    """
    Ternary quantization {-1,0,1} using fixed percentiles per column.
    The smallest weights become -1, the largest become +1, the rest are 0.
    """
    p_neg = 1 - p_zero - p_pos
    w_q = np.zeros_like(w)
    n_rows, n_cols = w.shape

    for c in range(n_cols):
        col = w[:, c]
        sorted_idx = np.argsort(col)  # indices of sorted weights
        n_neg = int(p_neg * n_rows)
        n_zero = int(p_zero * n_rows)

        w_q[sorted_idx[:n_neg], c] = -1        # smallest weights
        w_q[sorted_idx[n_neg:n_neg+n_zero], c] = 0
        w_q[sorted_idx[n_neg+n_zero:], c] = 1  # largest weights

    return w_q

def ternarize_threshold(w, k):
    """
    Ternary quantization {-1,0,1} using a column-wise threshold.
    Weights larger than k * mean(|w|) become ±1, others become 0.
    """
    w_q = np.zeros_like(w)
    n_rows, n_cols = w.shape

    for c in range(n_cols):
        col = w[:, c]
        t = k * np.mean(np.abs(col))  # threshold for this column
        w_q[:, c] = np.where(col > t, 1, np.where(col < -t, -1, 0))  # ternary mapping

    return w_q



def example_direct(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    svm = LinearSVC(C=Config.SVM_C, max_iter=10000)
    svm.fit(x_train, y_train)

    #Logistic Regression
    logreg = LogisticRegression(
        solver="lbfgs",
        max_iter=10000
    )
    logreg.fit(x_train, y_train)

    print("svm direct accuracy", svm.score(x_test, y_test))
    print("logreg direct accuracy", logreg.score(x_test, y_test))

def compare_performance(x_test, y_test, svm, logreg):
    #-------------------------Testing the svm on SNPS-------------------------
    snps_svm_pred, _, svm_scores = extend_and_test(x_test,"svm", svm.coef_, None)
    snps_svm_accuracy = np.mean(snps_svm_pred == y_test)
    print("SNPS svm accuracy:", snps_svm_accuracy)

    #add importance
    snps_imp_svm_pred, features, svm_imp_scores = extend_and_test(x_test,"svm_imp", svm.coef_, get_importance(svm.coef_))
    snps_imp_svm_accuracy = np.mean(snps_imp_svm_pred == y_test)
    print("SNPS svm imp accuracy:", snps_imp_svm_accuracy)

    features_int_pos = np.maximum(features // 4, 0) #extract feature from images

    #real weights SVM
    raw_svm_accuracy = svm.score(features_int_pos.T, y_test)
    print("real weights svm accuracy:", raw_svm_accuracy)

    #-------------------------Testing the logreg on SNPS-------------------------
    snps_lr_pred, _, lr_scores = extend_and_test(x_test,"lr", logreg.coef_, None)
    snps_lr_accuracy = np.mean(snps_lr_pred == y_test)
    print("SNPS logreg accuracy:", snps_lr_accuracy)

    #add importance
    snps_imp_lr_pred, _, lr_imp_scores = extend_and_test(x_test,"lr_imp", logreg.coef_, get_importance(logreg.coef_))
    snps_imp_lr_accuracy = np.mean(snps_imp_lr_pred == y_test)
    print("SNPS logreg imp accuracy:", snps_imp_lr_accuracy)

    #real weights LOGREG
    raw_lr_accuracy = logreg.score(features_int_pos.T, y_test)
    print("real weights logreg accuracy:", raw_lr_accuracy)

    #------------------combined charge------------
    # sum charge of the models without importance
    sum_pred = svm_scores.T + lr_scores.T
    sum_labels = np.argmax(sum_pred, axis=0)
    ens_accuracy = np.mean(sum_labels == y_test)
    print("SNPS ensemble accuracy:", ens_accuracy)

    #------------------combined IMP charge------------
    # sum charge of the models with importance (same as the SNPS with ensemble)
    sum_imp_pred = svm_imp_scores.T + lr_imp_scores.T
    sum_imp_labels = np.argmax(sum_imp_pred, axis=0)
    ens_imp_accuracy = np.mean(sum_imp_labels == y_test)
    print("SNPS imp ensemble accuracy:", ens_imp_accuracy)


    return snps_svm_accuracy, snps_lr_accuracy, snps_imp_svm_accuracy, snps_imp_lr_accuracy, ens_accuracy, ens_imp_accuracy, raw_svm_accuracy, raw_lr_accuracy


def extend_and_test(x_test, method, w, multipliers):
    snps = SNPSystem(Config.TEST_SIZE, Config.TEST_SIZE + 5, True)
    snps.spike_train = x_test
    q = ternarize_matrix(w.T)
    extended_path = extend_csv("csv/" + Config.CSV_NAME, np.array(q), method, multipliers)
    snps.load_neurons_from_csv(extended_path)
    snps.start()
    y_pred = np.argmax(snps.charge_map_prediction, axis=0)

    return y_pred, snps.pooling_image, snps.charge_map_prediction.T
