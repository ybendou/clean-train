import random 

class Args():
    def __init__(self):
        self.dataset = 'miniimagenet'
        self.runs = 1
        self.n_ways = 5
        self.n_queries = 15
        self.n_shots = [1,5]
        self.model = 'resnet12'
        self.output = ""
        self.feature_maps = 64
        self.rotations = False
        self.device = "cuda:0"
        self.devices = ["cuda:0"]
        self.mixup = False
        self.batch_size = 128
        self.dataset_device = "cuda:0"
        self.dataset_path = "../../../../datasets/"
        self.episodic = True
        self.episodes_per_epoch = 100
        self.seed = random.randint(0, 1000000000)
        self.deterministic = True
        self.ncm_loss = True
        self.n_runs = 10000
        self.lr = 0.1
        self.epochs = 0
        self.milestones = "100"
        self.gamma = -1
        self.cosine = True
        self.mixup = True
        self.dropout = 0
        self.preprocessing = ""
        self.manifold_mixup = 0
        self.temperature = 1
        self.skip_epochs = 0
        self.dataset_size = -1

        self.load_model = "../../../Experiments/models/vincent_backbones/resnet12standard.pt1"
        self.save_features = ""
        self.save_model = ""
        self.test_feature = ""















args = Args()
