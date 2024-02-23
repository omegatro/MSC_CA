import random
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso

def ca_transform(input_list, g=3, k=2, function=lambda x: max(x)):
    """
    Transform the input list to a lower-dimensional representation using a CA-like method.
    
    Parameters:
    - input_list: The list of numbers to be transformed.
    - g: The number of dimensions to keep in the final output.
    - k: The neighborhood size for the CA transformation.
    
    Returns:
    - A lower-dimensional representation of the input list.
    """
    if k <= 1:
        print('Please set k > 1. Using k+1 instead.')
        k += 1
    if k >= (len(input_list) - 1) // 2:
        print('Warning: k values >= (len(input_list) - 1)//2 will restrict the resulting number of dimensions to len(input_list) - k')
    
    m = (len(input_list) - g) / (k - 1)
    step_counter = 0
    numbers = input_list.copy()
    new_numbers = []
    
    while True:
        step_counter += 1
        if step_counter == round(m + 1) or step_counter == len(input_list):
            break
        
        for i in range(len(numbers) - k + 1):
            new_numbers.append(function(numbers[i:i+k]))
        
        numbers = new_numbers
        new_numbers = []
    
    return numbers


def revert_ca_emb(input, numbers):
    # Assuming V_ca and V are already defined
   V_ca = np.array([numbers])  # Reduced-dimension representation
   V = np.array([input])      # Original vector

   # Solve for X
   X, _, _, _ = np.linalg.lstsq(V_ca, V, rcond=None)

   # Use X to reconstruct v from v_ca, demonstrating the solution
   v_reconstructed = np.dot(V_ca, X)

   # Calculate reconstruction error (MSE)
   reconstruction_error = np.mean((V - v_reconstructed) ** 2)
   #Checking the actual difference
   print(f'Original: {V}', f'Reconstructed: {v_reconstructed}', f'Embedding: {V_ca}',sep='\n')#, f'Transformation:\n{X}'
   print("Reconstruction Error:", reconstruction_error)


def test_ca_embed(input_size=10, function=lambda x: max(x)):
   for n in range(2,(input_size-2)//2):
      for j in range(2,11):
         try:
            print(n,j)
            input = [random.randint(0,100) for _ in range(input_size)]
            numbers = ca_transform(input_list=input, k=n, g=j, function=function)
            revert_ca_emb(input, numbers)
         except Exception as e:
            print(n, j, e)


def ca_transform_matrix(input_matrix, g=3, k=2, function=lambda x: max(x)):
    """
    Transform each vector in the input matrix to a lower-dimensional representation using a CA-like method.
    
    Parameters:
    - input_matrix: A NumPy array where each row is a vector to be transformed.
    - g: The number of dimensions to keep for each vector in the final output.
    - k: The neighborhood size for the CA transformation.
    - function: The function applied to each neighborhood.
    
    Returns:
    - A NumPy array of lower-dimensional representations for each input vector.
    """
    transformed_vectors = []
    for input_list in input_matrix:
        # The transformation logic remains the same, applied per vector
        new_numbers = []
        numbers = input_list.copy()
        step_counter = 0
        m = (len(input_list) - g) / (k - 1)
        while True:
            step_counter += 1
            if step_counter == round(m + 1) or step_counter == len(input_list):
                break
            for i in range(len(numbers) - k + 1):
                new_numbers.append(function(numbers[i:i+k]))
            numbers = new_numbers
            new_numbers = []
        transformed_vectors.append(numbers)
    
    return np.array(transformed_vectors)


def revert_ca_matrix_emb(input_matrix, transformed_matrix):
    """
    Attempt to reconstruct the original matrix from the transformed matrix using least squares.
    
    Parameters:
    - input_matrix: The original matrix of vectors.
    - transformed_matrix: The matrix of transformed vectors.
    
    Returns:
    - Reconstruction error for the matrix.
    """
    # Ensure matrices are two-dimensional
    V_ca = transformed_matrix
    V = input_matrix
    print(V_ca.shape)
    print(V.shape)
    # Solve for X using least squares for each vector
    X, _, _, _ = np.linalg.lstsq(V_ca, V, rcond=None)  # Transpose to match dimensions
    
    # Reconstruct original matrix
    V_reconstructed = np.dot(V_ca, X)  # Transpose back after reconstruction

    # Calculate reconstruction error (MSE)
    reconstruction_error = np.mean((V - V_reconstructed) ** 2)
    print("Reconstruction Error:", reconstruction_error)
    #Checking the actual difference
    print(f'Original: {V}', f'Reconstructed: {V_reconstructed}', f'Embedding: {V_ca}', f'Transformation:\n{X}',sep='\n'), 
    print("Reconstruction Error:", reconstruction_error)
    return reconstruction_error


def reconstruct_with_sklearn(input_matrix, transformed_matrix, regularizer=None, alpha=1.0):
    """
    Reconstruct the original matrix from the transformed matrix using sklearn linear models,
    and return the reconstruction error along with the model's parameters.
    
    Parameters:
    - input_matrix: The original matrix of vectors.
    - transformed_matrix: The matrix of transformed vectors.
    - regularizer: The type of regularization to apply ('None', 'L2', or 'L1').
    - alpha: The regularization strength (applies only if a regularizer is selected).
    
    Returns:
    - Reconstruction error for the matrix.
    - Model coefficients (X matrix).
    - Model intercept.
    """
    # Select the model based on the regularizer option
    if regularizer == 'L2':
        model = Ridge(alpha=alpha)
    elif regularizer == 'L1':
        model = Lasso(alpha=alpha)
    else:  # Default to LinearRegression (no regularization)
        model = LinearRegression()
    
    # Fit the model to the transformed matrix and original matrix
    model.fit(transformed_matrix, input_matrix)
    
    # Predict/Reconstruct the original matrix
    V_reconstructed = model.predict(transformed_matrix)
    
    # Calculate reconstruction error (MSE)
    reconstruction_error = np.mean((input_matrix - V_reconstructed) ** 2)
    print(f"Reconstruction Error with {regularizer if regularizer else 'None'} Regularization:", reconstruction_error)
    print(f'Original:\n{input_matrix}', f'Reconstructed:\n{V_reconstructed}', f'Embedding:\n{transformed_matrix}', f'A:\n{model.coef_}', f'B:\n{model.intercept_}',sep='\n', ), 
    
    # Return the reconstruction error, model coefficients (X matrix), and intercept
    return reconstruction_error, model.coef_, model.intercept_


def test_ca_transform_matrix_and_revert(input_size=10, function=lambda x: max(x)):
    # Generate a matrix of random vectors
    input_matrix = np.array([random.sample(range(100), 10) for _ in range(input_size)])  # Example: 5 vectors of length 10

    for n in range(2,(input_size-2)//2):
      for j in range(2,input_size - 1):
         # Transform the matrix
         g = j  # Number of dimensions to keep
         k = n  # Neighborhood size

         try:
            print(f'Testing with k = {k}, g = {g}, number of columns = {input_size}')
            transformed_matrix = ca_transform_matrix(input_matrix, g, k, function=function)
            # Attempt to reconstruct the original matrix from the transformed matrix
            reconstruct_with_sklearn(input_matrix, transformed_matrix)
            reconstruct_with_sklearn(input_matrix, transformed_matrix, regularizer='L1')
            reconstruct_with_sklearn(input_matrix, transformed_matrix, regularizer='L2')
         except Exception as e:
            print(n, j, e)

if __name__ == "__main__":
   test_ca_transform_matrix_and_revert()
   
