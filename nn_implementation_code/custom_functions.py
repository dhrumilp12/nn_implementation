import torch
from torch.autograd import Function, gradcheck
# Identity function for testing
class IdentityFunction(Function):
    @staticmethod
    def forward(x):
        return x

    @staticmethod
    def backward(grad_output):
        return grad_output

# Sigmoid, Linear, CrossEntropy implementations
class SigmoidFunction(Function):
    @staticmethod
    def forward(ctx, input):
        """
        input: (N, D) or any shape
        output: sigmoid(input), same shape
        """
        sigmoid = 1.0 / (1.0 + torch.exp(-input))
        # Save output for backward (more stable)
        ctx.save_for_backward(sigmoid)
        return sigmoid

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: dL/d(sigmoid) with same shape as input
        returns: dL/d(input)
        """
        (sigmoid,) = ctx.saved_tensors
        grad_input = grad_output * sigmoid * (1.0 - sigmoid)
        return grad_input

# Linear function implementation
class LinearFunction(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias):
        """
        inp:    (N, in_features)
        weight: (out_features, in_features)
        bias:   (out_features,)
        output: (N, out_features)
        """
        output = inp.matmul(weight.t()) + bias
        ctx.save_for_backward(inp, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: dL/d(output) shape (N, out_features)
        returns:
            grad_inp:    dL/d(inp)    shape (N, in_features)
            grad_weight: dL/d(weight) shape (out_features, in_features)
            grad_bias:   dL/d(bias)   shape (out_features,)
        """
        inp, weight, bias = ctx.saved_tensors

        grad_inp = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_inp = grad_output.matmul(weight)  # (N, out) @ (out, in) -> (N, in)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(inp)  # (out, N) @ (N, in) -> (out, in)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)  # sum over batch

        return grad_inp, grad_weight, grad_bias

# Cross-entropy loss implementation
class CrossEntropyFunction(Function):
    @staticmethod
    def forward(ctx, logits, target):
        """
        logits: (N, K) raw scores (pre-softmax)
        target: (N,) class indices in [0, K-1]
        returns: scalar mean cross-entropy loss
        """
        # Numeric stability: log-softmax
        # subtract max per row
        max_logits, _ = logits.max(dim=1, keepdim=True)
        shifted = logits - max_logits  # (N, K)

        exp_shifted = shifted.exp()
        sum_exp = exp_shifted.sum(dim=1, keepdim=True)  # (N, 1)
        log_sum_exp = sum_exp.log()                      # (N, 1)

        # log-softmax
        log_probs = shifted - log_sum_exp               # (N, K)
        N = logits.shape[0]

        # Cross-entropy: -log p(target)
        loss = -log_probs[torch.arange(N), target].mean()

        # Softmax probabilities needed for gradient
        probs = exp_shifted / sum_exp  # (N, K)
        ctx.save_for_backward(probs, target)
        ctx.N = N

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output is dL_total/dLoss (usually 1.0 scalar)
        returns: grad_logits, None
        """
        probs, target = ctx.saved_tensors
        N = ctx.N

        grad_logits = probs.clone()  # (N, K)
        grad_logits[torch.arange(N), target] -= 1.0
        grad_logits /= N  # because we used mean over batch

        # Chain rule for possible upstream scaling
        grad_logits = grad_logits * grad_output

        return grad_logits, None


# Test the custom functions with gradcheck
if __name__ == "__main__":

    num = 4
    inp = 3

    x = torch.rand((num, inp), requires_grad=True).double()

    sigmoid = SigmoidFunction.apply

    assert gradcheck(sigmoid, x)
    print("Backward pass for sigmoid function is implemented correctly")

    out = 2

    x = torch.rand((num, inp), requires_grad=True).double()
    weight = torch.rand((out, inp), requires_grad=True).double()
    bias = torch.rand(out, requires_grad=True).double()

    linear = LinearFunction.apply
    assert gradcheck(linear, (x, weight, bias))
    print("Backward pass for linear function is implemented correctly")

    activations = torch.rand((15, 10), requires_grad=True).double()
    target = torch.randint(10, (15,))
    crossentropy = CrossEntropyFunction.apply
    assert gradcheck(crossentropy, (activations, target))
    print("Backward pass for crossentropy function is implemented correctly")
