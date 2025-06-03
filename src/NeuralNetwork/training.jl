module Training

using ..AutoDiff: GraphNode, Constant, Variable, topological_sort, forward!, backward!
using ..Layers: Chain, parameters
using ..Optimizers: Optimizer, update!

function _train_on_batch(model::Chain, order::Vector{GraphNode}, x_train_node::GraphNode, y_train_true_node::GraphNode, y_train_pred_node::GraphNode, accuracy_fn::Function, optimizer::Optimizer, x_batch::AbstractArray{<:Real}, y_batch::AbstractArray)::Tuple{Float32,Float32}
    x_train_node.output = x_batch
    y_train_true_node.output = y_batch

    loss, accuracy = _test_loss_and_accuracy(order, y_train_pred_node, accuracy_fn, y_batch)

    backward!(order)
    update!(optimizer, parameters(model))

    return loss, accuracy
end

function _test_loss_and_accuracy(order::Vector{GraphNode}, y_pred_node::GraphNode, accuracy_fn::Function, y_test::AbstractArray)::Tuple{Float32,Float32}
    loss = forward!(order)
    accuracy = accuracy_fn(y_pred_node.output, y_test)

    return loss, accuracy
end

function setup_training_functions(; model::Chain, loss_fn::Function, accuracy_fn::Function, optimizer::Optimizer, x_test::AbstractArray{T}, y_test::AbstractArray, batch_size::Int, epsilon::Float32=eps(Float32)) where T
    placeholder = Matrix{T}(undef, 0, 0)
    x_train_node = Variable(placeholder)
    y_train_true_node = Constant(placeholder)
    y_train_pred_node = model(x_train_node)
    train_loss_node = loss_fn(y_train_pred_node, y_train_true_node, batch_size=batch_size, epsilon=epsilon)
    train_order = topological_sort(train_loss_node)

    x_test_node = Variable(x_test)
    y_test_true_node = Constant(y_test)
    y_test_pred_node = model(x_test_node)
    test_loss_node = loss_fn(y_test_pred_node, y_test_true_node, batch_size=size(y_test, 2), epsilon=epsilon)
    test_order = topological_sort(test_loss_node)

    return (x_batch, y_batch) -> _train_on_batch(model, train_order, x_train_node, y_train_true_node, y_train_pred_node, accuracy_fn, optimizer, x_batch, y_batch),
    () -> _test_loss_and_accuracy(test_order, y_test_pred_node, accuracy_fn, y_test)
end

export setup_training_functions

end