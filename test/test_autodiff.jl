@testset "Autodiff" begin
    @testset "ScalarOperator" begin
        @testset "Power (^)" begin
            # Test 1
            x1 = Variable(2.0)
            n_const = Constant(3.0)
            z1 = x1^n_const
            order1 = topological_sort(z1)
            @test forward!(order1) ≈ 8.0
            backward!(order1)
            @test x1.gradient ≈ (3.0 * 2.0^(3.0 - 1.0))

            # Test 2
            x2 = Variable(2.0)
            n_var = Variable(3.0)
            z2 = x2^n_var
            order2 = topological_sort(z2)
            @test forward!(order2) ≈ 8.0
            backward!(order2)
            @test x2.gradient ≈ (3.0 * 2.0^(3.0 - 1.0))
            @test n_var.gradient ≈ (log(abs(2.0)) * 2.0^3.0)
        end

        @testset "Sin" begin
            x = Variable(0.5)
            s_op = sin(x)
            order = topological_sort(s_op)
            @test forward!(order) ≈ sin(0.5)
            backward!(order)
            @test x.gradient ≈ cos(0.5)
        end
    end

    @testset "BroadcastedOperator" begin
        @testset "Addition (.+)" begin
            # Test 1
            x1 = Variable([2.0, 3.0])
            x2 = Variable([4.0, 5.0])
            z1 = x1 .+ x2
            s1 = sum(z1) # Redukcja do skalara
            order1 = topological_sort(s1)
            @test forward!(order1) ≈ 2.0 + 4.0 + 3.0 + 5.0
            backward!(order1)
            @test x1.gradient ≈ [1.0, 1.0]
            @test x2.gradient ≈ [1.0, 1.0]

            # Test 2
            x3 = Variable([2.0, 3.0])
            n_const = Constant([1.0, 1.0])
            z2 = x3 .+ n_const
            s2 = sum(z2)
            order2 = topological_sort(s2)
            @test forward!(order2) ≈ 2.0 + 1.0 + 3.0 + 1.0
            backward!(order2)
            @test x3.gradient ≈ [1.0, 1.0]
        end

        @testset "Subtraction (.-)" begin
            x1 = Variable([5.0, 4.0])
            y1 = Variable([1.0, 3.0])
            z = x1 .- y1
            s = sum(z)
            order = topological_sort(s)
            @test forward!(order) ≈ 5.0 - 1.0 + 4.0 - 3.0
            backward!(order)
            @test x1.gradient ≈ [1.0, 1.0]
            @test y1.gradient ≈ [-1.0, -1.0]
        end

        @testset "Element-wise Multiplication (.*)" begin
            x1 = Variable([3.0, 2.0])
            x2 = Variable([4.0, 5.0])
            z = x1 .* x2
            s = sum(z)
            order = topological_sort(s)
            @test forward!(order) ≈ 3.0 * 4.0 + 2.0 * 5.0
            backward!(order)
            @test x1.gradient ≈ [4.0, 5.0]
            @test x2.gradient ≈ [3.0, 2.0]
        end

        @testset "Element-wise Division (./)" begin
            x1 = Variable([10.0, 20.0])
            x2 = Variable(2.0)
            z = x1 ./ x2
            s = sum(z)
            order = topological_sort(s)
            @test forward!(order) ≈ 10.0 / 2.0 + 20.0 / 2.0
            backward!(order)
            @test x1.gradient ≈ [1.0 / 2.0, 1.0 / 2.0]
            @test x2.gradient ≈ -10.0 / (2.0^2) + -20.0 / (2.0^2)
        end

        @testset "Matrix-Vector Multiplication (*)" begin
            A = Variable([1.0 2.0; 3.0 4.0])
            x = Variable([5.0, 6.0])
            z = A * x
            s = sum(z)
            order = topological_sort(s)
            @test forward!(order) ≈ 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 5.0 + 4.0 * 6.0
            backward!(order)
            @test A.gradient ≈ [5.0 6.0; 5.0 6.0]
            @test x.gradient ≈ [1.0 + 3.0, 2.0 + 4.0]
        end

        @testset "Sum" begin
            x = Variable([1.0, 2.0, 3.0, 4.0])
            s = sum(x)
            order = topological_sort(s)
            @test forward!(order) ≈ 1.0 + 2.0 + 3.0 + 4.0
            backward!(order)
            @test x.gradient ≈ [1.0, 1.0, 1.0, 1.0]
        end

        @testset "Max Element-wise (max.)" begin
            x1 = Variable([1.0, -2.0, 5.0])
            x2 = Variable([3.0, -1.0, 2.0])
            m = max.(x1, x2)
            s = sum(m)
            order = topological_sort(s)
            @test forward!(order) ≈ 3.0 + -1.0 + 5.0
            backward!(order)
            @test x1.gradient ≈ [0.0, 0.0, 1.0]
            @test x2.gradient ≈ [1.0, 1.0, 0.0]
        end

        @testset "Softmax (basic check)" begin
            x = Variable([1.0, 2.0, 0.5])
            softmax_ = softmax(x)

            order1 = topological_sort(softmax_)
            @test forward!(order1) ≈ exp.([1.0, 2.0, 0.5]) ./ sum(exp.([1.0, 2.0, 0.5]))

            s = sum(softmax_)
            order2 = topological_sort(s)
            @test forward!(order2) ≈ 1.0 # suma wyjściowych prawdopodobieństw powinna być równa 1
            backward!(order2)
            expected_gradient = [0.0, 0.0, 0.0] # suma wyjść softmaxa jest "płaska" względem zmian w jego wejściach
            @test x.gradient ≈ expected_gradient atol = 1e-16 # dla 0 '≈' jest jak '=' https://www.jlhub.com/julia/manual/en/function/isapprox
        end

        @testset "Log Element-wise (log.)" begin
            x1 = 2.0
            x2 = 4.0
            x = Variable([x1, x2])
            l = log.(x)
            s = sum(l)
            order = topological_sort(s)
            @test forward!(order) ≈ (log(x1) + log(x2))
            backward!(order)
            @test x.gradient ≈ [1 / x1, 1 / x2]
        end

        @testset "Sigmoid (σ)" begin
            x = Float32[0.5, -0.2, 0.0]
            x_sig = Variable(x)
            z = σ(x_sig)
            s = sum(z)
            order = topological_sort(s)
            expected_forward_result = 1.0f0 ./ (1.0f0 .+ exp.(-x))
            @test forward!(order) ≈ sum(expected_forward_result)
            backward!(order)
            @test x_sig.gradient ≈ (expected_forward_result .* (1.0f0 .- expected_forward_result))
        end
    end

    @testset "Combined Operations" begin
        @testset "Simple Chain: σ(sum(w .* x) .+ b)" begin
            w = Variable([2.0, 3.0])
            x = Variable([0.5, 0.1])
            bias = Variable(1.0)

            mul = w .* x
            sum_mul = sum(mul)
            affine_scalar = sum_mul .+ bias
            sigmoid_output = σ(affine_scalar)

            order = topological_sort(sigmoid_output)

            mul_val = (0.5 * 2.0) + (0.1 * 3.0)
            affine_val = mul_val + 1.0
            sigmoid_output_val = 1 / (1 + exp(-affine_val))
            @test forward!(order) ≈ sigmoid_output_val

            backward!(order)

            grad_affine_local = sigmoid_output_val * (1 - sigmoid_output_val) # σ'(affine_scalar)
            @test bias.gradient ≈ grad_affine_local
            @test w.gradient ≈ grad_affine_local .* [0.5, 0.1]
            @test x.gradient ≈ grad_affine_local .* [2.0, 3.0]
        end
    end
end