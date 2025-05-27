using Pkg
Pkg.activate(".")
Pkg.instantiate()


using JLD2
X_train = load("./data/embeddings/imdb_dataset_prepared.jld2", "X_train")
y_train = load("./data/embeddings/imdb_dataset_prepared.jld2", "y_train")
X_test = load("./data/embeddings/imdb_dataset_prepared.jld2", "X_test")
y_test = load("./data/embeddings/imdb_dataset_prepared.jld2", "y_test")
embeddings = load("./data/embeddings/imdb_dataset_prepared.jld2", "embeddings")
vocab = load("./data/embeddings/imdb_dataset_prepared.jld2", "vocab")
nothing

# Print data variable information
println("X_train: size = ", size(X_train), ", first few values = ", X_train[1:5])
println("y_train: size = ", size(y_train), ", first few values = ", y_train[1:5])
println("X_test: size = ", size(X_test), ", first few values = ", X_test[1:5])
println("y_test: size = ", size(y_test), ", first few values = ", y_test[1:5])
println("embeddings: size = ", size(embeddings), ", first few values = ", embeddings[1:5,1:5])
println("vocab: length = ", length(vocab), ", first few values = ", vocab[1:5])

embedding_dim = size(embeddings, 1)



using AWID.NeuralNetwork, AWID.AutoDiff

model = Chain(
    Embedding(length(vocab), embedding_dim),
    Permute((2,1,3)),
    Conv((3,), embedding_dim, 8, relu),
    MaxPool((8,)),
    Flatten(),
    Dense(128, 1, sigmoid)
)

# add Glove embeddings to Embedding layer
model.layers[1].W.output .= embeddings;


using AWID.NeuralNetwork, AWID.AutoDiff, Printf

batch_size = 64

train_on_batch, test_loss_and_accuracy = setup_training_functions(
    model=model,
    loss_fn=binary_crossentropy,
    accuracy_fn=accuracy,
    optimizer=Adam(),
    x_test=X_test,
    y_test=y_test,
    batch_size=batch_size,
)

epochs = 5
for epoch in 1:epochs
    epoch_total_loss = 0.0f0
    epoch_total_acc = 0.0f0
    num_processed_batches = 0

    epoch_batches = get_epoch_batches(X_train, y_train, batch_size=batch_size, do_shuffle=true)

    t = @elapsed begin
        for (x_batch, y_batch) in epoch_batches
            batch_loss, batch_acc = train_on_batch(x_batch, y_batch)

            epoch_total_loss += batch_loss
            epoch_total_acc += batch_acc
            num_processed_batches += 1
        end

        train_loss = epoch_total_loss / num_processed_batches
        train_acc = epoch_total_acc / num_processed_batches

        test_loss, test_acc = test_loss_and_accuracy()
    end

    println(@sprintf("Epoch: %d (%.2fs) \tTrain: (l: %.2f, a: %.2f) \tTest: (l: %.2f, a: %.2f)",
        epoch, t, train_loss, train_acc, test_loss, test_acc))
end