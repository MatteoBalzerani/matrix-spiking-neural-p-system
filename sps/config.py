class Config:
    MODE = "CNN"
    DATABASE = 'digit'
    NUM_LAYERS = 4

    # IMAGES
    IMG_SHAPE = 28 #ipotizing squared shape images of 28 pixels
    BLOCK_SHAPE = 2  #the size of the window block for the second layer

    #SNPS BEHAVIOUR
    INVERT = False #invert or not invert the spike in the starting images
    QUANTIZATION = True
    WHITE_HOLE= True #if true all the internal spikes are deleted after firing

    TRAIN_SIZE = 5000
    TEST_SIZE = 2500

    #LAYER 1 - INPUT IMAGE
    NEURONS_L1 = int(IMG_SHAPE ** 2) #number of neurons for layer 1 (pixels in the image)
    Q_RANGE = 6  # the range of quantization, it works on images, rules and tuning

    #LAYER 2 - FEATURE EXTRACTION
    KERNELS = [
        [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
        [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]],
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        [[-1, -1, 0], [-1, 0, 1], [0, 1, 1]],
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
        [[0, -1, -1], [1, 0, -1], [1, 1, 0]],
        [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
        [[1, 1, 0], [1, 0, -1], [0, -1, -1]]
    ]
    KERNEL_NUMBER = len(KERNELS) #number of kernels in layer 2
    KERNEL_SHAPE = len(KERNELS[0]) #size of a single kernel
    SHAPE_FEATURE = int(IMG_SHAPE - KERNEL_SHAPE + 1) #size of a generated image in layer 2 (26)
    NEURONS_FEATURE = int(SHAPE_FEATURE ** 2) #number of neurons for each image in layer 2 (676)
    NEURONS_L2 = NEURONS_FEATURE * KERNEL_NUMBER #number of neurons for layer 2 (5408)

    #LAYER 3 - AVERAGE POOLING
    POOLING_SIZE = 2 #size of the pooling window
    SHAPE_POOL = int(SHAPE_FEATURE / POOLING_SIZE) #size of the resulting image after the pooling (13)
    NEURONS_POOL = int(SHAPE_POOL ** 2) #number of neurons for each image in layer 3 (169)
    NEURONS_L3 = int(KERNEL_NUMBER * NEURONS_POOL) #number of neurons on the pooling layer (1352)

    #LAYER 3 - CLASSIFICATION
    CLASSES = 10
    SVM_C = 3.0

    #MATRIX TERNARIZE
    TERNARIZE_METHOD = 2 #can be 1 -> Percentile or 2 -> Threshold
    M_SPARSITY = 0.8 #percentage of 0 values in the quantized matrix, used if TERNARIZE_METHOD == 1
    M_POSITIVE = 0.1 #percentage of 1 values in the quantized matrix, used if TERNARIZE_METHOD == 1
    M_THRESHOLD = 1.05 #multiplied factor for column values, used if TERNARIZE_METHOD == 2

    #IMPORTANCE
    IMPORTANCE_METHOD = 2 #how the model calculate the magnitude of the weights
    DISCRETIZE_METHOD = 2 #how the model apply the importance to rules
    DISC_RANGE = 4  #range of importance values, used if DISCRETIZE_METHOD == 2

    # ENERGY COSTS
    WORST_REGEX = 100
    EXPECTED_REGEX = 10
    EXPECTED_SPIKE = 0.5

    THRESHOLD = 128 # higher Thr -> more spike

    CSV_NAME = "SNPS.csv"
    CSV_EXACT_NAME = "SNPS_exact.csv"
    CSV_EXACT_ENS_NAME = "SNPS_exact_ens.csv"
    CSV_ENS_NAME = "SNPS_ens.csv"
    CSV_CNN_NAME = "SNPS_cnn.csv"

    # Charge tracker output integration (Francesca)
    TRACK_CHARGES = False
    TRACK_MODE = "step_by_step" # "step_by_step" saves after each image, "all_at_once" saves at the end of the execution
    TRACK_FORMAT = "csv"      # "csv" or "parquet"
    TRACK_FILENAME = "output_charges"

    TIME_TEST_NUM = 0

    @classmethod
    def compute_k_range(cls):
        cls.K_RANGE = [
            (
                sum(v == -1 for row in kernel for v in row) * (-cls.Q_RANGE + 1),
                sum(v == 1  for row in kernel for v in row) *  (cls.Q_RANGE - 1)
            )
            for kernel in cls.KERNELS
        ]


def database(database):
    Config.DATABASE = database

    if database == "tissuemnist":
        Config.IMG_SHAPE = 28
        Config.CLASSES = 8

    if database == "breastmnist":
        Config.IMG_SHAPE = 28
        Config.CLASSES = 2
        if Config.TRAIN_SIZE > 546: Config.TRAIN_SIZE = 546
        if Config.TRAIN_SIZE > 156: Config.TRAIN_SIZE = 156

    if database == "octmnist":
        Config.IMG_SHAPE = 28
        Config.CLASSES = 4

    if database == "bloodmnist":
        Config.IMG_SHAPE = 28
        Config.CLASSES = 8

    if database == "pathmnist":
        Config.IMG_SHAPE = 28
        Config.CLASSES = 9

    if database == "dermamnist":
        Config.IMG_SHAPE = 28
        Config.CLASSES = 7

    if database == "flower":
        Config.IMG_SHAPE = 224
        Config.CLASSES = 102
        Config.TRAIN_SIZE = 2040
        Config.TEST_SIZE = 1020

    if database == "digit":
        Config.IMG_SHAPE = 28
        Config.CLASSES = 10

    # Keep derived architecture fields aligned after dataset-specific changes.
    Config.KERNEL_NUMBER = len(Config.KERNELS)
    Config.KERNEL_SHAPE = len(Config.KERNELS[0])
    Config.SHAPE_FEATURE = int(Config.IMG_SHAPE - Config.KERNEL_SHAPE + 1)
    Config.NEURONS_FEATURE = int(Config.SHAPE_FEATURE ** 2)
    Config.NEURONS_L2 = Config.NEURONS_FEATURE * Config.KERNEL_NUMBER
    Config.SHAPE_POOL = int(Config.SHAPE_FEATURE / Config.POOLING_SIZE)
    Config.NEURONS_POOL = int(Config.SHAPE_POOL ** 2)
    Config.NEURONS_L3 = int(Config.KERNEL_NUMBER * Config.NEURONS_POOL)

