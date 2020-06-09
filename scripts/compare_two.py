import argparse
import pandas as pd
import evaluate


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to input file.")
    parser.add_argument("--output_file_1", type=str, help="Path to output file 1.")
    parser.add_argument("--output_file_2", type=str, help="Path to output file 2.")
    parser.add_argument("--golden_file", type=str, help="Path to file with golden values.")
    args = parser.parse_args()

    inp = pd.read_csv(args.input_file).to_numpy()
    out1 = pd.read_csv(args.output_file_1).to_numpy()
    out2 = pd.read_csv(args.output_file_2).to_numpy()
    gold = pd.read_csv(args.golden_file).to_numpy()
    
    print(evaluate.compare(inp, out1, out2, gold))
