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
    # Debugging of errors:
    #it = np.nditer(inp, flags=['multi_index', 'refs_ok'])
    #while not it.finished:
    #    if inp[it.multi_index] == '?' and output[it.multi_index] != golden_output[it.multi_index]:
    #        print('prediction', output[it.multi_index], '     !=     gold', golden_output[it.multi_index])
    #    it.iternext()
    #print('correctly predicted', predicted_right, 'out of', total)
    return predicted_right / total 

def compare(inp, output1, output2, golden_output):
    should_predict = (inp == '?')
    total = np.sum(should_predict)
    predicted_right_1 = 0
    predicted_right_2 = 0
    predicted_right_any = 0
    for i in range(len(should_predict)):
        for j in range(len(should_predict[i])):
            if should_predict[i,j]:
                if output1[i,j] == golden_output[i,j]:
                    predicted_right_1 += 1
                if output2[i,j] == golden_output[i,j]:
                    predicted_right_2 += 1
                if output1[i,j] == golden_output[i,j] or output2[i,j] == golden_output[i,j]:
                    predicted_right_any += 1
    
    return (predicted_right_1 / total, predicted_right_2 / total, predicted_right_any / total)

if __name__ == "__main__":
    import pandas as pd
    train_X = pd.read_csv('../data/train_x.csv').to_numpy()
    train_y = pd.read_csv('../data/train_y.csv').to_numpy()

    output = train_y.copy()
    output[:,70:] = train_X[:,70:]
    print(evaluate(train_X, output, train_y))
