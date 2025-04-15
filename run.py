import easydict
import yaml
import argparse
from easydict import EasyDict
from script.run_synthetic import run_synthetic

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='ZoAR')
    parser.add_argument('--config', type=str, default='config/synthetic.yaml', help='Path to the config file')
    path_to_config = parser.parse_args().config
    with open(path_to_config, 'r') as f:
        args = yaml.safe_load(f)
    args = EasyDict(args)

    assert args.exp in ["synthetic", "cutest", "adversarial", "fine-tuning"]
    if args.exp == "synthetic":
        run_synthetic(args)
    elif args.exp == "cutest":
        pass
    elif args.exp == "adversarial":
        pass
    elif args.exp == "fine-tuning":
        pass

if __name__ == '__main__':
    main()
