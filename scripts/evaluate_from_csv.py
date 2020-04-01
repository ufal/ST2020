import argparse
import pandas as pd
import evaluate


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to input file.")
    parser.add_argument("--output_file", type=str, help="Path to output file.")
    parser.add_argument("--golden_file", type=str, help="Path to file with golden values.")
    args = parser.parse_args()

    inp = pd.read_csv(args.input_file).to_numpy()
    out = pd.read_csv(args.output_file).to_numpy()
    gold = pd.read_csv(args.golden_file).to_numpy()
    print("Accuracy is {:.2%}".format(evaluate.evaluate(inp, out, gold)))