# rand(Float32, dims...) -> losowa macierz Float32 o podanych wymiarach
# .* (2f0 * scale) .- scale -> skaluje zakres [0, 1) do [-scale, scale)

function init_xavier_glorot(in_dim, out_dim)
    scale = sqrt(6f0 / (in_dim + out_dim))
    return rand(Float32, out_dim, in_dim) .* 2f0 .* scale .- scale
end

function init_xavier_glorot_conv1d(filter_width::Int, in_channels::Int, out_channels::Int)
    fan_in = filter_width * in_channels # liczba wag na wejściu do pojedynczego filtra
    fan_out = filter_width * out_channels # liczba wag wychodzących
    scale = sqrt(6f0 / (fan_in + fan_out))
    return rand(Float32, filter_width, in_channels, out_channels) .* (2f0 * scale) .- scale
end

function init_zeros(dims...)
    return zeros(Float32, dims...)
end
