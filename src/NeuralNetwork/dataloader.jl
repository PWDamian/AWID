module DataLoader

using Random: shuffle

function get_epoch_batches(X_data::AbstractMatrix{T}, Y_data::AbstractMatrix{S}; batch_size::Int, do_shuffle::Bool=false) where {T,S}
    num_samples = size(X_data, 2) # cechy(1) x próbki(2)
    indices = 1:num_samples

    if do_shuffle
        indices = shuffle(indices)
    end

    num_batches_for_epoch = ceil(Int, num_samples / batch_size)

    return ( # To jest generator, który zwraca iterator, który zwraca batche
        (
            X_data[:, indices[(i-1)*batch_size+1:min(i * batch_size, num_samples)]],
            Y_data[:, indices[(i-1)*batch_size+1:min(i * batch_size, num_samples)]]
        )
        for i in 1:num_batches_for_epoch
    )
end

export get_epoch_batches

end