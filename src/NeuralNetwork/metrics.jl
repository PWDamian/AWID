module Metrics

using Statistics: mean

function accuracy(y_pred_values::AbstractArray, y_true_values::AbstractArray)
    predicted_labels = y_pred_values .> 0.5f0
    true_labels = y_true_values .> 0.5f0

    return mean(predicted_labels .== true_labels)
end

export accuracy

end