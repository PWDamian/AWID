# funkcje pomocnicze
update!(node::Constant, gradient) = nothing
update!(node::GraphNode, gradient) =
    if isnothing(node.gradient)
        node.gradient = gradient # pierwszy gradient dla tego węzła
    else
        node.gradient = node.gradient .+ gradient # (.+) - broadcast/elementwise
    end

function backward!(order::Vector; seed::Float32=1.0f0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

# Constant i Variable to zawsze liście (pozostałe węzły to zawsze Operator), na liściach się zatrzymujemy, dlatego nothing
function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    gradients = backward(node, extract_input_values(node)..., node.gradient)
    for (input, gradient) in zip(node.inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end