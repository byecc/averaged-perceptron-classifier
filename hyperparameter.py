class Parameter:
    def __init__(self):
        self.learn_rate = 0.1
        self.batch_size = 50
        self.iter_num = 512
        self.kthi = 0.001
        self.class_num = 5
        self.clean_switch = True  #是否清洗数据

        self.ap_iter_num = 1000
        self.depth = 0
        self.ap_batch_size = 45
        # self.test_interval = 256
        # self.threshold = 0


        self.lamda = 0.1
        self.interval_batch_size = 48


        # self.model_path = "F:\jcode\englishPCFG.ser.gz"  #stanford
        self.model_path = "F:\jcode\englishFactored.ser.gz"  #stanford
        self.graph_dir = "dp_graph2"
        self.tree_path = "data/tree.pkl"
        self.tree_height = 2
        self.features_dir = "features_file"
        self.dict_dir = "dict_file"
