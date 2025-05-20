@testset "NeuralNetwork" begin
    @testset "Layers" begin
        d1 = Dense(2, 3, relu)
        d2 = Dense(3, 1, sigmoid)
        model = Chain(d1, d2)

        @testset "Dense Layer" begin
            @test d1.W.output isa Matrix{Float32}
            @test size(d1.W.output) == (3, 2)
            @test d1.b.output isa Vector{Float32}
            @test size(d1.b.output) == (3,) # size zwraca jednoelementową krotkę, dlatego (3,) a nie 3
            @test d1.activation_fn == relu

            @test size(d2.W.output) == (1, 3)
            @test size(d2.b.output) == (1,)
        end

        @testset "Chain" begin
            @test length(model.layers) == 2
            @test model.layers[1] == d1
            @test model.layers[2] == d2
        end

        @testset "Parameters" begin
            params_d1 = parameters(d1)
            @test length(params_d1) == 2
            @test params_d1[1] === d1.W
            @test params_d1[2] === d1.b

            params_model = parameters(model)
            @test length(params_model) == 4
            @test params_model[1] === d1.W
            @test params_model[2] === d1.b
            @test params_model[3] === d2.W
            @test params_model[4] === d2.b
        end

        @testset "Dense Layer Forward Pass" begin
            W = Float32[1.0 -2.0; -3.0 4.0; 0.5 -0.5]
            b = Float32[-0.1, 0.2, 0.6]

            dense_layer = Dense(Variable(W), Variable(b), relu) # pomijamy inicjalizację losowych wag

            input = Float32[1.0, 2.0]
            input_node = Variable(input)

            output_node = dense_layer(input_node)
            order = topological_sort(output_node)
            forward_result = forward!(order)

            # max.([1*1+(-2)*2; -3*1+4*2; 0.5*1+(-0.5)*2] .+ [-0.1;0.2;0.6], 0.0f0) = max.([-3.1; 5.2; 0.1], 0.0f0)
            expected_result = [0.0, 5.2, 0.1] # W * input .+ b 
            @test forward_result ≈ expected_result
        end

        @testset "Chain Forward Pass" begin
            W1 = Float32[1.0 0.0; 0.0 1.0]
            b1 = Float32[0.1, -0.1]
            d1 = Dense(Variable(W1), Variable(b1), relu)

            W2 = Float32[2.0 0.0]
            b2 = Float32[0.5]
            d2 = Dense(Variable(W2), Variable(b2), relu)

            model = Chain(d1, d2)

            input = Float32[1.0, 2.0]
            input_node = Variable(input)

            output_node = model(input_node)
            order = topological_sort(output_node)
            forward_result = forward!(order)

            # d1_out = W1*input + b1 = [1 0; 0 1]*[1;2] + [0.1;-0.1] = [1.1; 1.9]
            # d2_out = W2*d1_out + b2 = [2 0]*[1.1; 1.9] + [0.5] = [2.7]
            @test forward_result ≈ [2.7f0]
        end
    end

    @testset "Losses" begin
        @testset "BinaryCrossEntropy" begin
            y_hat = 0.6899745f0 # Przykładowe prawdopodobieństwo
            y_true = 1.0f0

            y_hat_node = Variable(y_hat)
            y_true_node = Constant(y_true) # Constant, bo nie podlega optymalizacji (stały cel)

            epsilon = eps(Float32)
            loss_node = binary_crossentropy(y_hat_node, y_true_node, batch_size=1, epsilon=epsilon) # batch_size=1, bo test dla 1 próbki
            order = topological_sort(loss_node)
            loss = forward!(order)

            expected_loss = -(y_true * log(y_hat + epsilon) + (1.0f0 - y_true) * log(1.0f0 - y_hat + epsilon))
            @test loss ≈ expected_loss
        end
    end

    @testset "Optimizers" begin
        @testset "Adam" begin
            param = Float32[1.0, 2.0]
            param_var = Variable(param)
            gradient = Float32[0.1, -0.2] # Przykładowy gradient
            param_var.gradient = gradient
            opt = Adam(lr=0.1f0)
            params_list = [param_var]

            m1 = (1 - opt.beta1) * gradient
            v1 = (1 - opt.beta2) * (gradient .^ 2)
            m1_hat = m1 / (1 - opt.beta1^1)
            v1_hat = v1 / (1 - opt.beta2^1)
            expected_new_param = param .- opt.lr .* m1_hat ./ (sqrt.(v1_hat) .+ opt.epsilon) # obliczenia przed update!, bo nadpisuje 

            update!(opt, params_list)

            @test param_var.output ≈ expected_new_param
        end
    end
end