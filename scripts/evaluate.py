import numpy as np

def evaluate(inp, output, golden_output):
    """
    Expectes three numpy arrays, where 
    inp is input matrix with ? in it, so script knows which values were predicted
    output is matrix of all values (even those that were not predicted)
    golden_ouput is 2D matrix of all golden values (even those that were not predicted)
    """
    should_predict = (inp == '?')
    total = np.sum(should_predict)
    predicted_right = np.sum(output[should_predict] == golden_output[should_predict])
    print('correctly predicted', predicted_right, 'out of', total)
    return predicted_right / total 


if __name__ == "__main__":
    import pandas as pd
    train_X = pd.read_csv('../data/train_x.csv').to_numpy()
    train_y = pd.read_csv('../data/train_y.csv').to_numpy()

    output = train_y.copy()
    output[:,70:] = train_X[:,70:]
    print(evaluate(train_X, output, train_y))
