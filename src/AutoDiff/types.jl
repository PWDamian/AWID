import Base: show # import potrzebny, bo dodajemy własne metody show do funkcji z innego modułu

abstract type GraphNode end
abstract type Operator <: GraphNode end # reprezentuje operacje matematyczne (<: = dziedziczenie)

struct Constant{T} <: GraphNode
    output::T # gradient = 0
end

mutable struct Variable <: GraphNode
    output::Any # input = output
    gradient::Any
    name::String
    Variable(output; name="?") = new(output, nothing, name) # constructor
end

mutable struct ScalarOperator{F} <: Operator
    inputs::Any # krotka węzłów GraphNode, które są wejściami dla tej operacji
    output::Any # wynik operacji podczas przejścia w przód
    gradient::Any # przechowuje gradient wyniku względem siebie propagowany z góry
    name::String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs::Any
    output::Any
    gradient::Any
    name::String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
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