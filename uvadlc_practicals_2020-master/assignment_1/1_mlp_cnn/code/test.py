import numpy as np

test = np.array([[-7.64148003e-05, 1.04828724e-05, 2.54159197e-04, 1.03560681e-04, 2.02121806e-04, 6.38651218e-05, -2.07595991e-04, -7.61438108e-05, -2.08101719e-04, 3.26954680e-04]
, [2.97750535e-04, -4.81460616e-05, 1.24425867e-05, -3.22430530e-05, 1.48463879e-04, -8.18636797e-05, -1.15632414e-04, -1.86714389e-04, 2.07177066e-04, 4.73346428e-04]
, [-2.79682333e-04, 1.24883180e-04, 4.96087367e-05, -8.57175464e-06, -3.45131090e-05, 3.88719771e-05, -1.81303827e-04, -4.13441326e-04, 8.02591445e-05, 5.22694345e-05]])

predictions = np.array([[0.25, 0.25, 0.4, 0.05, 0.05],
                        [0.8, 0.05, 0.05, 0.05, 0.05],
                        [0.1, 0.5, 0.1, 0.1, 0.1],
                        [0.01, 0.09, 0.7, 0.1, 0.1],
                        [0.2, 0, 0.2, 0.6, 0]])

targets = np.array([[1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0]])

def accuracy(predictions, targets):
    preds = np.argmax(predictions, axis=1)
    labels = np.argmax(targets, axis=1)
    acc = np.mean(preds == labels)
    print(acc)

accuracy(predictions, targets)

# def softmax1(z):
#     e = np.exp(z-np.max(z))
#     s = np.sum(e, axis=1, keepdims=True)
#     out = e/s
#     return out

# def softmax2(z):
#     e = np.exp(z-np.max(z, axis=0))
#     s = np.sum(e, axis=1, keepdims=True)
#     out = e/s
#     return out

# print(np.sum(softmax1(test)))
# print(np.sum(softmax2(test)))


def batchnorm_forward(x, gamma, beta, eps):

  N, D = x.shape

  #step1: calculate mean
  mu = 1./N * np.sum(x, axis = 0)

  #step2: subtract mean vector of every trainings example
  xmu = x - mu

  #step3: following the lower branch - calculation denominator
  sq = xmu ** 2

  #step4: calculate variance
  var = 1./N * np.sum(sq, axis = 0)

  #step5: add eps for numerical stability, then sqrt
  sqrtvar = np.sqrt(var + eps)

  #step6: invert sqrtwar
  ivar = 1./sqrtvar

  #step7: execute normalization
  xhat = xmu * ivar

  #step8: Nor the two transformation steps
  gammax = gamma * xhat

  #step9
  out = gammax + beta

  #store intermediate
  cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

  return out, cache


  def batchnorm_backward(dout, cache):

  #unfold the variables stored in cache
  xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache

  #get the dimensions of the input/output
  N,D = dout.shape

  #step9
  dbeta = np.sum(dout, axis=0)
  dgammax = dout #not necessary, but more understandable

  #step8
  dgamma = np.sum(dgammax*xhat, axis=0)
  dxhat = dgammax * gamma

  #step7
  divar = np.sum(dxhat*xmu, axis=0)
  dxmu1 = dxhat * ivar

  #step6
  dsqrtvar = -1. /(sqrtvar**2) * divar

  #step5
  dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

  #step4
  dsq = 1. /N * np.ones((N,D)) * dvar

  #step3
  dxmu2 = 2 * xmu * dsq

  #step2
  dx1 = (dxmu1 + dxmu2)
  dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

  #step1
  dx2 = 1. /N * np.ones((N,D)) * dmu

  #step0
  dx = dx1 + dx2

  return dx, dgamma, dbeta