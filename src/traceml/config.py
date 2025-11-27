class TraceMLConfig:

    def __init__(self):
        self.enable_logging: bool = False
        self.logs_dir: str = "./logs"
        self.num_display_layers = 10


config = TraceMLConfig()
