import Base: show # import potrzebny, bo dodajemy własne metody show do funkcji z innego modułu

abstract type GraphNode end
abstract type Operator <: GraphNode end # reprezentuje operacje matematyczne (<: = dziedziczenie)

mutable struct Constant{T} <: GraphNode
    output::T # gradient = 0
end

mutable struct Variable{T} <: GraphNode
    output::T
    gradient::Union{T,Nothing}
    name::String
    Variable(output::T, name::String="?") where T = new{T}(output, nothing, name)
end

mutable struct ScalarOperator{F,I<:Tuple{Vararg{GraphNode}}} <: Operator
    inputs::I # krotka węzłów GraphNode, które są wejściami dla tej operacji
    output::Union{Float32,Nothing} # wynik operacji podczas przejścia w przód
    gradient::Union{Float32,Nothing} # przechowuje gradient wyniku względem siebie propagowany z góry
    name::String
    ScalarOperator(fun::F, inputs::GraphNode...; name::String="?") where {F} = new{F,typeof(inputs)}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F,I<:Tuple{Vararg{GraphNode}}} <: Operator
    inputs::I
    output::Union{AbstractArray{Float32},Float32,Nothing}
    gradient::Union{AbstractArray{Float32},Float32,Nothing}
    name::String
    BroadcastedOperator(fun::F, inputs::GraphNode...; name::String="?") where {F} = new{F,typeof(inputs)}(inputs, nothing, nothing, name)
end

show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")")
show(io::IO, x::BroadcastedOperator{F,I}) where {F,I} = print(io, "op.", x.name, "(", F, ")")
show(io::IO, x::Constant{T}) where {T} = print(io, "const ", x.output)
show(io::IO, x::Variable{T}) where {T} = begin
    print(io, "var ", x.name)
    print(io, "\n ┣━ ^ ")
    summary(io, x.output)
    print(io, "\n ┗━ ∇ ")
    summary(io, x.gradient)
end