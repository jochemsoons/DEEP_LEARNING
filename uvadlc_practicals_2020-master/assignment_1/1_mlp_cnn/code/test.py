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