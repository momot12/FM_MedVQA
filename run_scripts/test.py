from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("model")
parser.add_argument("metric")
parser.add_argument("dataset")
parser.add_argument("output_dir")
args = parser.parse_args()


if args.model == 'llava':
    pass
elif args.model == 'vilt':
    pass
elif args.model == 'llava-lora':
    pass
elif args.model == 'vilt-lora':
    pass
elif args.model == 'med-llava':
    pass
elif args.model == 'bio-vilt':
    pass


# TODO: larger pretrained model minigpt-v