#include "fftconv2d.h"

at::Tensor complex_matmul(at::Tensor a, at::Tensor b);

at::Tensor conv2d(at::Tensor signal, at::Tensor kernel, int64_t groups) {
    // perform rfftn on x and y
    auto signal_fr = rfftn(signal);
    auto kernel_fr = rfftn(kernel);

    kernel_fr = at::conj(kernel_fr);

    // signal_fr is of shape (batch_size, in_channels, height, width)
    // view signal_fr as (batch_size, groups, in_channels/groups, height, width)
    // hard-code now, should be replace with inference later
    signal_fr = signal_fr.view({signal_fr.size(0), groups, signal_fr.size(1) / groups, signal_fr.size(2), signal_fr.size(3)});

    // kernel_fr is of shape (out_channels, in_channels/groups, height, width)
    // view kernel_fr as (groups, outchannel/groups, in_channels/groups, height, width)
    // hard-code now, should be replace with inference later
    kernel_fr = kernel_fr.view({groups, kernel_fr.size(0) / groups, kernel_fr.size(1), kernel_fr.size(2), kernel_fr.size(3)});

    // movedim signal_fr to (batch_size, groups, height, width, 1, in_channels/groups)
    signal_fr = torch::movedim(signal_fr, 2, signal_fr.dim() - 1).unsqueeze(signal_fr.dim() - 1);
    // movedim kernel_fr to (1, groups, height, width, in_channels/groups, outchannel/groups)
    kernel_fr = torch::movedim(kernel_fr, {1, 2}, {kernel_fr.dim() - 1, kernel_fr.dim() - 2});
    // ! matmul only support float32
    auto result = complex_matmul(signal_fr, kernel_fr);
    result = torch::movedim(result, result.dim() - 1, 2).squeeze(result.dim() - 1);
    // move dim to (batch_size, out_channels, height, width)
    result = result.view({result.size(0), result.size(1) * result.size(2), result.size(3), result.size(4)});
    return result;
    // perform irfftn on result
    // return irfftn(result); 
}

at::Tensor complex_matmul(at::Tensor a, at::Tensor b) {
    auto a_real = at::real(a);
    auto a_imag = at::imag(a);
    auto b_real = at::real(b);
    auto b_imag = at::imag(b);

    auto real = torch::empty(a.sizes(), a.options());
    auto imag = torch::empty(a.sizes(), a.options());

    real = torch::matmul(a_real, b_real) - torch::matmul(a_imag, b_imag);
    imag = torch::matmul(a_real, b_imag) + torch::matmul(a_imag, b_real);

    auto ret = at::complex(real, imag);
    return ret;
}