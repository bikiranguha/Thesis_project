# Function to get the approximate entropy of a time series, by specifying the forward steps and filtering levels
# Approximate entropy: Measure of the unpredictability of the time series, higher the entropy, higher the unpredictability
# Link for more info: https://en.wikipedia.org/wiki/Approximate_entropy 
import numpy as np

def ApEn(U, m, r):
    # U: Any sequential data, m: No. of time steps ahead to look at for unpredictability
    # r: filtering level (threshold which defines what is a reasonable difference between 2 data points)

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m+1) - _phi(m))


if __name__ == '__main__':
    # Usage example
    U = np.array([85, 80, 89] * 17)
    print ApEn(U, 2, 3)
    1.0996541105257052e-05

    randU = np.random.choice([85, 80, 89], size=17*3)
    print (ApEn(randU, 2, 3))