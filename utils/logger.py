import os
import os.path as osp
import time
import atexit

# Try to import wandb (optional dependency)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class Logger:
    """
    A general-purpose logger.
    adapted from https://github.com/openai/spinningup

    Makes it easy to save diagnostics, hyperparameter configurations, the
    state of a training run, and the trained model.
    """
    def __init__(self, output_dir=None, output_fname='progress.txt', exp_name=None,
                 use_wandb=True, wandb_project=None, wandb_entity=None, config=None):
        """
        Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.

            output_fname (string): Name for the tab-separated-value file
                containing metrics logged throughout a training run.
                Defaults to ``progress_exp_name.txt``.

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)

            use_wandb (bool): Whether to log to Weights & Biases. Defaults to True.
                Set to False to disable wandb logging.

            wandb_project (string): W&B project name. If None, uses WANDB_PROJECT env var.

            wandb_entity (string): W&B entity (team/user). If None, uses WANDB_ENTITY env var.

            config (dict or object): Configuration to log to W&B.
        """
        self.exp_name = exp_name
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
        atexit.register(self.output_file.close)
        print(colorize("Logging data to {}".format(self.output_file.name), 'green', bold=True))
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}

        # Initialize wandb if available and requested
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_run = None

        if self.use_wandb:
            # Get project and entity from environment or parameters
            project = wandb_project or os.environ.get('WANDB_PROJECT', 'POCO-TTT')
            entity = wandb_entity or os.environ.get('WANDB_ENTITY', 'neuroai')

            # Convert config to dict if it's an object
            config_dict = None
            if config is not None:
                if hasattr(config, '__dict__'):
                    config_dict = {k: v for k, v in config.__dict__.items()
                                   if not k.startswith('_') and not callable(v)}
                elif isinstance(config, dict):
                    config_dict = config

            # Generate standardized run name
            run_name = self._generate_run_name(exp_name, config_dict)

            try:
                self.wandb_run = wandb.init(
                    project=project,
                    entity=entity,
                    name=run_name,
                    group=exp_name,  # Group runs by experiment name
                    config=config_dict,
                    dir=output_dir,
                    reinit=True,
                )
                print(colorize(f"Logging to W&B: {project}/{run_name}", 'cyan', bold=True))
            except Exception as e:
                print(colorize(f"Failed to initialize W&B: {e}", 'yellow'))
                self.use_wandb = False

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers,\
                "Trying to introduce a new key {} that you didn't include in the first iteration".format(key)
        assert key not in self.log_current_row,\
            "You already set {} this iteration. Maybe you forgot to call dump_tabular()".format(key)
        self.log_current_row[key] = val

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = '%' + '%d' % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
            vals.append(val)
        print("-" * n_slashes, flush=True)

        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()

        # Log to wandb if available
        if self.use_wandb and self.wandb_run is not None:
            # Create a dict of metrics to log
            wandb_metrics = {}
            for key in self.log_headers:
                val = self.log_current_row.get(key)
                if val is not None and hasattr(val, "__float__"):
                    wandb_metrics[key] = float(val)
            if wandb_metrics:
                wandb.log(wandb_metrics)

        self.log_current_row.clear()
        self.first_row = False

    def finish(self):
        """Finish logging and close wandb run."""
        if self.use_wandb and self.wandb_run is not None:
            wandb.finish()
            self.wandb_run = None

    def _generate_run_name(self, exp_name: str, config_dict: dict) -> str:
        """
        Generate a standardized run name for wandb.

        Naming schema: {mode}_{model}_{dataset}_s{seed}

        Where:
        - mode: Training mode (standard/fomaml/e2e_ttt/single)
        - model: Model label (POCO, NLinear, MLP, etc.)
        - dataset: Dataset label (celegansflavell, zebrafishahrens_pc, etc.)
        - seed: Random seed number

        Examples:
        - standard_POCO_celegansflavell_s0
        - fomaml_POCO_celegansflavell_s0
        - e2e_ttt_POCO_zebrafishahrens_pc_s1
        - single_POCO_celegansflavell-0_s0
        """
        if config_dict is None:
            return exp_name

        # Extract components from config
        model = config_dict.get('model_label', 'unknown')
        dataset = config_dict.get('dataset_label', 'unknown')
        seed = config_dict.get('seed', 0)
        training_mode = config_dict.get('training_mode', 'standard')

        # Handle list datasets (multi-session training)
        if isinstance(dataset, list):
            if len(dataset) == 1:
                dataset = dataset[0]
            else:
                # Abbreviate multiple datasets
                dataset = '+'.join([d.split('_')[0][:4] for d in dataset[:3]])
                if len(config_dict.get('dataset_label', [])) > 3:
                    dataset += '+...'

        # Determine mode from experiment name if not in config
        if training_mode == 'standard':
            if 'fomaml' in exp_name.lower():
                training_mode = 'fomaml'
            elif 'e2e_ttt' in exp_name.lower():
                training_mode = 'e2e_ttt'
            elif 'single_session' in exp_name.lower() or '-' in str(dataset):
                training_mode = 'single'

        # Shorten mode names for readability
        mode_short = {
            'standard': 'std',
            'fomaml': 'fom',
            'e2e_ttt': 'e2e',
            'single': 'ss',
        }.get(training_mode, training_mode[:3])

        # Shorten dataset names for readability
        dataset_short = str(dataset)
        dataset_short = dataset_short.replace('celegansflavell', 'cef')
        dataset_short = dataset_short.replace('zebrafishahrens_pc', 'zfa_pc')
        dataset_short = dataset_short.replace('zebrafishahrens', 'zfa')
        dataset_short = dataset_short.replace('zebrafish_pc', 'zf_pc')
        dataset_short = dataset_short.replace('zebrafish', 'zf')
        dataset_short = dataset_short.replace('mice_pc', 'mice_pc')
        dataset_short = dataset_short.replace('celegans', 'cel')

        # Build run name
        run_name = f"{mode_short}_{model}_{dataset_short}_s{seed}"

        return run_name
