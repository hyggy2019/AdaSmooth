import easydict
import yaml
import argparse
from easydict import EasyDict
from script.run_synthetic import run_synthetic
from script.run_cutest import run_cutest
from script.run_adversarial import run_adversarial

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='ZoAR')
    parser.add_argument('--config', type=str, default='config/synthetic.yaml', help='Path to the config file')
    path_to_config = parser.parse_args().config
    with open(path_to_config, 'r') as f:
        args = yaml.safe_load(f)
    args = EasyDict(args)

    if args.exp == "synthetic":
        run_synthetic(args)
    elif args.exp == "cutest":
        run_cutest(args)
    elif args.exp == "adversarial":
        run_adversarial(args)
    elif args.exp == "fine-tuning":
        pass
    else:
        raise ValueError(f"Unknown experiment type: {args.exp}. Available options are: synthetic, cutest, adversarial, fine-tuning")

if __name__ == '__main__':
    main()
