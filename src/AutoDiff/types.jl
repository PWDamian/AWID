import Base: show # import potrzebny, bo dodajemy własne metody show do funkcji z innego modułu

abstract type GraphNode end
abstract type Operator <: GraphNode end # reprezentuje operacje matematyczne (<: = dziedziczenie)

mutable struct Constant{T} <: GraphNode
    output::T # gradient = 0
end

mutable struct Variable{T} <: GraphNode
    output::T
    gradient::Any
    name::String
    Variable(output::T, name="?") where T = new{T}(output, nothing, name)
end

mutable struct ScalarOperator{F,I<:Tuple{Vararg{GraphNode}}} <: Operator
    inputs::I # krotka węzłów GraphNode, które są wejściami dla tej operacji
    output::Any # wynik operacji podczas przejścia w przód
    gradient::Any # przechowuje gradient wyniku względem siebie propagowany z góry
    name::String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun),Tuple{Vararg{GraphNode}}}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F,I<:Tuple{Vararg{GraphNode}}} <: Operator
    inputs::I
    output::Union{AbstractArray{Float32},Number,Nothing}
    gradient::Union{AbstractArray{Float32},Number,Nothing}
    name::String
    BroadcastedOperator(fun::Function, inputs_tuple::Vararg{GraphNode}; name::String="?") = new{typeof(fun),typeof(inputs_tuple)}(inputs_tuple, nothing, nothing, name)
end

show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")")
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")")
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "var ", x.name)
    print(io, "\n ┣━ ^ ")
    summary(io, x.output)
    print(io, "\n ┗━ ∇ ")
    summary(io, x.gradient)
end