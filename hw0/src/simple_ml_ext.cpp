#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <cstring>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */
    const float *X_batch;
    const unsigned char *y_batch;
    float gradient_at_i_and_j;

    float *Z = (float *) malloc(sizeof(float) * (batch * k));
    if (Z == nullptr) exit(1);
    float *Iy = (float *) malloc(sizeof(float) * (batch * k));
    if (Iy == nullptr) exit(1);
    float *row_wise_sum_for_normalization = (float *) malloc(sizeof(float) * batch);
    if (row_wise_sum_for_normalization == nullptr) exit(1);

    for (size_t example_index = 0; example_index < (m / batch) * batch; example_index += batch) {
        X_batch = X + example_index * n;
        y_batch = y + example_index;

        //=========== calculate Iy : start
        // set Iy[i, y_batch[i]] to 1, and all other elements are set to zero
        memset(Iy, 0, sizeof(float) * (batch * k));
        for (size_t i = 0; i < batch; i++)
        {
            Iy[i * k + y_batch[i]] = 1;
        }
        //=========== calculate Iy : end


        //========calculate Z : start
        memset(Z, 0, sizeof(float) * (batch * k));
        // caculate exp(matmul(X_batch, theta)), shape=(batch, k)
        // X_batch[i, t] * theta[t, j]
        for (size_t i = 0; i < batch; i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                for (size_t t = 0; t < n; t++)
                {
                    Z[i * k + j] += X_batch[i * n + t] * theta[t * k + j]; 
                }
                Z[i * k + j] = exp(Z[i * k + j]);
            }
        }
        // normalize Z row-wise, shape=(batch, k)
        memset(row_wise_sum_for_normalization, 0, sizeof(float) * batch);
        for (size_t i = 0; i < batch; i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                row_wise_sum_for_normalization[i] += Z[i * k + j];
            }
            for (size_t j = 0; j < k; j++)
            {
                Z[i * k + j] /= row_wise_sum_for_normalization[i];
            }
        }
        //=========calculate Z : end



        //=========== modify  theta : start 
        // modify theta[i, j], theta.shape = (n, k)
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                // calculate gradient[i, j]
                // X_batch[t, i] * (Z - Iy)[t, j]
                gradient_at_i_and_j = 0;
                for (size_t t = 0; t < batch; t++)
                {
                    gradient_at_i_and_j += X_batch[t * n + i] * 
                        (Z[t * k + j] - Iy[t * k + j]);
                }
                gradient_at_i_and_j /= batch;

                // modify theta[i, j]
                theta[i * k + j] -= lr * gradient_at_i_and_j;
            }
        } 
        //=========== modify  theta : end
    }

    free(Z);
    free(Iy);
    free(row_wise_sum_for_normalization);
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
