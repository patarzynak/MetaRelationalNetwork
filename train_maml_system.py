from data import AlmostClevrMetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import OurMAMLFewShotClassifier
from utils.parser_utils import get_args
from utils.dataset_tools import maybe_unzip_dataset

# Combines the arguments, model, data and experiment builders to run an experiment
args, device = get_args()
model = OurMAMLFewShotClassifier(args=args, device=device,
                                 im_shape=(2, args.image_channels, args.image_height, args.image_width))
#maybe_unzip_dataset(args=args)
data = AlmostClevrMetaLearningSystemDataLoader
maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
maml_system.run_experiment()
