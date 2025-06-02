@testset "Autodiff" begin
    @testset "ScalarOperator" begin
        @testset "Power (^)" begin
            # Test 1
            x1 = Variable(2.0f0)
            n_const = Constant(3.0f0)
            z1 = x1^n_const
            order1 = topological_sort(z1)
            @test forward!(order1) ≈ 8.0f0
            backward!(order1)
            @test x1.gradient ≈ (3.0f0 * 2.0f0^(3.0f0 - 1.0f0))

            # Test 2
            x2 = Variable(2.0f0)
            n_var = Variable(3.0f0)
            z2 = x2^n_var
            order2 = topological_sort(z2)
            @test forward!(order2) ≈ 8.0f0
            backward!(order2)
            @test x2.gradient ≈ (3.0f0 * 2.0f0^(3.0f0 - 1.0f0))
            @test n_var.gradient ≈ (log(abs(2.0f0)) * 2.0f0^3.0f0)
        end

        @testset "Sin" begin
            x = Variable(0.5f0)
            s_op = sin(x)
            order = topological_sort(s_op)
            @test forward!(order) ≈ sin(0.5f0)
            backward!(order)
            @test x.gradient ≈ cos(0.5f0)
        end
    end

    @testset "BroadcastedOperator" begin
        @testset "Addition (.+)" begin
            # Test 1
            x1 = Variable([2.0f0, 3.0f0])
            x2 = Variable([4.0f0, 5.0f0])
            z1 = x1 .+ x2
            s1 = sum(z1) # Redukcja do skalara
            order1 = topological_sort(s1)
            @test forward!(order1) ≈ 2.0f0 + 4.0f0 + 3.0f0 + 5.0f0
            backward!(order1)
            @test x1.gradient ≈ [1.0f0, 1.0f0]
            @test x2.gradient ≈ [1.0f0, 1.0f0]

            # Test 2
            x3 = Variable([2.0f0, 3.0f0])
            n_const = Constant([1.0f0, 1.0f0])
            z2 = x3 .+ n_const
            s2 = sum(z2)
            order2 = topological_sort(s2)
            @test forward!(order2) ≈ 2.0f0 + 1.0f0 + 3.0f0 + 1.0f0
            backward!(order2)
            @test x3.gradient ≈ [1.0f0, 1.0f0]
        end

        @testset "Subtraction (.-)" begin
            x1 = Variable([5.0f0, 4.0f0])
            y1 = Variable([1.0f0, 3.0f0])
            z = x1 .- y1
            s = sum(z)
            order = topological_sort(s)
            @test forward!(order) ≈ 5.0f0 - 1.0f0 + 4.0f0 - 3.0f0
            backward!(order)
            @test x1.gradient ≈ [1.0f0, 1.0f0]
            @test y1.gradient ≈ [-1.0f0, -1.0f0]
        end

        @testset "Element-wise Multiplication (.*)" begin
            x1 = Variable([3.0f0, 2.0f0])
            x2 = Variable([4.0f0, 5.0f0])
            z = x1 .* x2
            s = sum(z)
            order = topological_sort(s)
            @test forward!(order) ≈ 3.0f0 * 4.0f0 + 2.0f0 * 5.0f0
            backward!(order)
            @test x1.gradient ≈ [4.0f0, 5.0f0]
            @test x2.gradient ≈ [3.0f0, 2.0f0]
        end

        @testset "Element-wise Division (./)" begin
            x1 = Variable([10.0f0, 20.0f0])
            x2 = Variable([2.0f0, 3.0f0])
            z = x1 ./ x2
            s = sum(z)
            order = topological_sort(s)
            @test forward!(order) ≈ 10.0f0 / 2.0f0 + 20.0f0 / 3.0f0
            backward!(order)
            @test x1.gradient ≈ [1.0f0 / 2.0f0, 1.0f0 / 3.0f0]
            @test x2.gradient ≈ [-10.0f0 / (2.0f0^2), -20.0f0 / (3.0f0^2)]
        end

        @testset "Matrix-Vector Multiplication (*)" begin
            A = Variable([1.0f0 2.0f0; 3.0f0 4.0f0])
            x = Variable([5.0f0, 6.0f0])
            z = A * x
            s = sum(z)
            order = topological_sort(s)
            @test forward!(order) ≈ 1.0f0 * 5.0f0 + 2.0f0 * 6.0f0 + 3.0f0 * 5.0f0 + 4.0f0 * 6.0f0
            backward!(order)
            @test A.gradient ≈ [5.0f0 6.0f0; 5.0f0 6.0f0]
            @test x.gradient ≈ [1.0f0 + 3.0f0, 2.0f0 + 4.0f0]
        end

        @testset "Sum" begin
            x = Variable([1.0f0, 2.0f0, 3.0f0, 4.0f0])
            s = sum(x)
            order = topological_sort(s)
            @test forward!(order) ≈ 1.0f0 + 2.0f0 + 3.0f0 + 4.0f0
            backward!(order)
            @test x.gradient ≈ [1.0f0, 1.0f0, 1.0f0, 1.0f0]
        end

        @testset "Max Element-wise (max.)" begin
            x1 = Variable([1.0f0, -2.0f0, 5.0f0])
            x2 = Variable([3.0f0, -1.0f0, 2.0f0])
            m = max.(x1, x2)
            s = sum(m)
            order = topological_sort(s)
            @test forward!(order) ≈ 3.0f0 + -1.0f0 + 5.0f0
            backward!(order)
            @test x1.gradient ≈ [0.0f0, 0.0f0, 1.0f0]
            @test x2.gradient ≈ [1.0f0, 1.0f0, 0.0f0]
        end

        @testset "Softmax" begin
            x = Variable([1.0f0, 2.0f0, 0.5f0])
            softmax_ = softmax(x)

            order1 = topological_sort(softmax_)
            @test forward!(order1) ≈ exp.([1.0f0, 2.0f0, 0.5f0]) ./ sum(exp.([1.0f0, 2.0f0, 0.5f0]))

            s = sum(softmax_)
            order2 = topological_sort(s)
            @test forward!(order2) ≈ 1.0f0 # suma wyjściowych prawdopodobieństw powinna być równa 1
            backward!(order2)
            expected_gradient = [0.0f0, 0.0f0, 0.0f0] # suma wyjść softmaxa jest "płaska" względem zmian w jego wejściach
            @test x.gradient ≈ expected_gradient atol = 1e-7 # dla 0 '≈' jest jak '=' https://www.jlhub.com/julia/manual/en/function/isapprox
        end

        @testset "Log Element-wise (log.)" begin
            x1 = 2.0f0
            x2 = 4.0f0
            x = Variable([x1, x2])
            l = log.(x)
            s = sum(l)
            order = topological_sort(s)
            @test forward!(order) ≈ (log(x1) + log(x2))
            backward!(order)
            @test x.gradient ≈ [1 / x1, 1 / x2]
        end

        @testset "Sigmoid (σ)" begin
            x = [0.5f0, -0.2f0, 0.0f0]
            x_sig = Variable(x)
            z = σ(x_sig)
            s = sum(z)
            order = topological_sort(s)
            expected_forward_result = 1.0f0 ./ (1.0f0 .+ exp.(-x))
            @test forward!(order) ≈ sum(expected_forward_result)
            backward!(order)
            @test x_sig.gradient ≈ (expected_forward_result .* (1.0f0 .- expected_forward_result))
        end

        @testset "embedding_lookup" begin
            embedding_weights = Variable([
                1f0 2f0 3f0;
                4f0 5f0 6f0
            ]) # 2 cechy (embedding_dim) dla 3 tokenów
            input_indices = Variable([
                1 2; # token 1
                2 3 # token 2
            ]) # 2 tokeny (w poziomie), 2 próbki (batch size=2, w pionie)
            node = embedding_lookup(embedding_weights, input_indices)
            input_values = AutoDiff.extract_input_values(node)
            expected = Array{Float32,3}(undef, 2, 2, 2)
            expected[:, 1, 1] = [1f0, 4f0]
            expected[:, 2, 1] = [2f0, 5f0]
            expected[:, 1, 2] = [2f0, 5f0]
            expected[:, 2, 2] = [3f0, 6f0]
            @test AutoDiff.forward(node, input_values...) ≈ expected
            expected_gradient = [1f0 2f0 1f0; 1f0 2f0 1f0] # kolumna 1 wybrana raz, kolumna 2 dwa razy, kolumna 3 raz
            gradient = AutoDiff.backward(node, input_values..., ones(Float32, 2, 2, 2))
            @test gradient[1] ≈ expected_gradient
        end

        @testset "conv1d" begin
            input_node = Variable(reshape([1f0, 2f0, 3f0, 4f0, 5f0], 1, 5, 1))
            weights_node = Variable(reshape([1f0, 0f0, -1f0], 3, 1, 1))
            bias_node = Variable([0f0])
            node = conv1d(input_node, weights_node, bias_node)
            input_values = AutoDiff.extract_input_values(node)
            expected = Array{Float32,3}(undef, 1, 3, 1)
            expected[1, 1, 1] = -2f0 # 1*1+2*0+3*(-1)=-2
            expected[1, 2, 1] = -2f0 # 2*1+3*0+4*(-1)=-2
            expected[1, 3, 1] = -2f0 # 3*1+4*0+5*(-1)=-2
            @test AutoDiff.forward(node, input_values...) ≈ expected
            gradient = AutoDiff.backward(node, input_values..., ones(Float32, 1, 3, 1))
            expected_dx = Array{Float32,3}(undef, 1, 5, 1)
            expected_dx[1, 1, 1] = 1f0 # 1*1 = 1
            expected_dx[1, 2, 1] = 1f0 # 1*1 + 1*0 = 1
            expected_dx[1, 3, 1] = 0f0 # 1*1 + 1*0 + 1*(-1) = 0
            expected_dx[1, 4, 1] = -1f0 # 1*(-1) + 1*0 = -1
            expected_dx[1, 5, 1] = -1f0 # 1*(-1) = -1
            @test gradient[1] ≈ expected_dx
            expected_dw = Array{Float32,3}(undef, 3, 1, 1)
            expected_dw[1, 1, 1] = 6f0 # 1*1 + 2*1 + 3*1 = 6
            expected_dw[2, 1, 1] = 9f0 # 2*1 + 3*1 + 4*1 = 9
            expected_dw[3, 1, 1] = 12f0 # 3*1 + 4*1 + 5*1 = 12
            @test gradient[2] ≈ expected_dw
            expected_db = [3f0] # 1 + 1 + 1 = 3
            @test gradient[3] ≈ expected_db
        end

        @testset "maxpool1d" begin
            input_node = Variable(reshape([1f0, 3f0, 2f0, 4f0], 1, 4, 1))
            pool_size_node = Constant(2)
            node = maxpool1d(input_node, pool_size_node)
            input_values = AutoDiff.extract_input_values(node)
            expected = Array{Float32,3}(undef, 1, 2, 1)
            expected[1, 1, 1] = 3f0
            expected[1, 2, 1] = 4f0
            @test AutoDiff.forward(node, input_values...) ≈ expected
            gradient = AutoDiff.backward(node, input_values..., ones(Float32, 1, 2, 1))
            expected_grad = zeros(Float32, 1, 4, 1)
            expected_grad[1, 2, 1] = 1f0
            expected_grad[1, 4, 1] = 1f0
            @test gradient[1] ≈ expected_grad
        end

        @testset "flatten" begin
            input_node = Variable(reshape(Float32[1 2 3; 4 5 6], 2, 3, 1))
            node = flatten(input_node)
            input_values = AutoDiff.extract_input_values(node)
            expected = Array{Float32,2}(undef, 6, 1)
            expected[:, 1] = [1f0, 4f0, 2f0, 5f0, 3f0, 6f0]
            @test AutoDiff.forward(node, input_values...) ≈ expected
            gradient = AutoDiff.backward(node, input_values..., ones(Float32, 6, 1))
            @test gradient[1] ≈ fill(1f0, 2, 3, 1)
        end
    end

    @testset "Combined Operations" begin
        @testset "Simple Chain: σ(sum(w .* x) .+ b)" begin
            w = Variable([2.0f0, 3.0f0])
            x = Variable([0.5f0, 0.1f0])
            bias = Variable(1.0f0)

            mul = w .* x
            sum_mul = sum(mul)
            affine_scalar = sum_mul .+ bias
            sigmoid_output = σ(affine_scalar)

            order = topological_sort(sigmoid_output)

            mul_val = (0.5f0 * 2.0f0) + (0.1f0 * 3.0f0)
            affine_val = mul_val + 1.0f0
            sigmoid_output_val = 1.0f0 / (1.0f0 + exp(-affine_val))
            @test forward!(order) ≈ sigmoid_output_val

            backward!(order)

            grad_affine_local = sigmoid_output_val * (1f0 - sigmoid_output_val)
            @test bias.gradient ≈ grad_affine_local
            @test w.gradient ≈ grad_affine_local .* [0.5f0, 0.1f0]
            @test x.gradient ≈ grad_affine_local .* [2.0f0, 3.0f0]
        end
    end
end