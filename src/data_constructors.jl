using Distributions
using Random
using DataFrames
using CSV
using DataStructures
using JuMP
using OrderedCollections
using LinearAlgebra


############################################################################################### 
################################## Auumptions ################################################

### The power generators should be ordered and a int value should be assigned to each generator
### The bus should be ordered and a int value should be assigned to each bus

########################### Data Constructors #################################################
###############################################################################################
# Define a struct to represent a power generator
struct PowerGenerator
    generator_no::Int
    start_up_cost::Float64
    shut_down_cost::Float64
    constant_cost_coefficient::Float64
    linear_cost_coefficient::Float64
    Min_electricty_output_limit::Float64
    Max_electricty_output_limit::Float64
    Min_up_time::Float64
    Min_down_time::Float64
    Ramp_up_limit::Float64
    Ramp_down_limit::Float64
    Start_up_ramp_rate_limit::Float64
    Shut_down_ramp_rate_limit::Float64
    bus_no::Int
end

# Constructor for PowerGenerator from a DataFrame row, assuming the DataFrame has named columns
function PowerGenerator(row::DataFrameRow)
    return PowerGenerator(
        row.generator_no,
        row.start_up_cost,
        row.shut_down_cost,
        row.constant_cost_coefficient,
        row.linear_cost_coefficient,
        row.Min_electricty_output_limit,
        row.Max_electricty_output_limit,
        row.Min_up_time,
        row.Min_down_time,
        row.Ramp_up_limit,
        row.Ramp_down_limit,
        row.Start_up_ramp_rate_limit,
        row.Shut_down_ramp_rate_limit,
        row.bus_no
    )
end

### Power_Generator_Set takes input and reads the data from the dataframe or matrix and each row is a power generator with its parameters
### The function returns a dictionary of power generators
### The number of power generators is equal to the number of rows in the dataframe or matrix
### The number of columns in the dataframe or matrix is equal to the number of parameters of the power generator

#### The input data must be a DataFrame or a Matrix
#### The columns of the input data must be in the following order:
#### start_up_cost, shut_down_cost, Min_up_time, Min_down_time, Min_electricty_output_limit, Max_electricty_output_limit,
#### Ramp_up_limit, Ramp_down_limit, Start_up_ramp_up_rate_limit, Shut_down_ramp_down_rate_limit
#### The number of columns must be equal to the number of parameters of the power generator

# Function to construct a set of generators from a DataFrame with named columns
function PowerGenerator_Set(df::DataFrame)
    # Define the required columns
    required_cols = [:generator_no, :start_up_cost, :shut_down_cost, 
                     :constant_cost_coefficient, :linear_cost_coefficient,
                     :Min_electricty_output_limit, :Max_electricty_output_limit,
                     :Min_up_time, :Min_down_time,
                     :Ramp_up_limit, :Ramp_down_limit,
                     :Start_up_ramp_rate_limit, :Shut_down_ramp_rate_limit,
                     :bus_no]
    
    # Check that all required columns are present
    missing_cols = setdiff(required_cols, names(df))
    if !isempty(missing_cols)
        error("Missing required columns: $(missing_cols)")
    end

    generators = Dict{Int, PowerGenerator}()

    for row in eachrow(df)
        gen = PowerGenerator(row)
        generators[gen.generator_no] = gen
    end
    return OrderedDict(sort(collect(generators)))
end


###############################################################################################



"""
    group_generators_by_bus(generator_col::Vector{Int}, bus_col::Vector{Int})
    Inputs: 
    - generator_col: A vector of generator numbers
    - bus_col: A vector of bus numbers
    Output:
    - A dictionary mapping bus numbers to vectors of generator numbers
    Example:
    group_generators_by_bus([1, 2, 3, 4], [1, 2, 1, 2]) -> OrderedDict{Int, Vector{Int}}(1 => [1, 3], 2 => [2, 4])
"""
function group_generators_by_bus(generator_col::Vector{Int}, bus_col::Vector{Int})
    @assert length(generator_col) == length(bus_col) "The generator and bus columns must be the same length."

    # Initialize an empty dictionary mapping bus numbers to vectors of generator numbers
    bus_dict = Dict{Int, Vector{Int}}()

    # Loop over each pair of generator and bus numbers
    for (generator_no, bus_no) in zip(generator_col, bus_col)
        if haskey(bus_dict, bus_no)
            push!(bus_dict[bus_no], generator_no)
        else
            bus_dict[bus_no] = [generator_no]
        end
    end

    # Return an OrderedDict sorted by bus numbers
    return OrderedDict(sort(collect(bus_dict)))
end


###############################################################################################

# Define a custom struct to represent an edge's properties
struct EdgeProperties
    edge_no::Int
    susceptance::Float64
    min_capacity::Float64
    max_capacity::Float64
    node_tuple::Tuple{Int, Int}  # or use appropriate types if nodes are not Int
end

# Constructor from a DataFrame row
function EdgeProperties(row::DataFrameRow)
    return EdgeProperties(
        row.edge_no,
        row.susceptance,
        row.min_capacity,
        row.max_capacity,
        (row.node1, row.node2)
    )
end

# For DataFrame input (assuming the DataFrame has the proper column names)

function edge_properties_set(df::DataFrame)
    # Validate required columns exist
    required = [:edge_no, :susceptance, :min_capacity, :max_capacity, :node1, :node2]
    missing = setdiff(required, names(df))
    if !isempty(missing)
        error("DataFrame is missing columns: $(missing)")
    end

    edge_dict = Dict{Int, EdgeProperties}()
    for row in eachrow(df)
        ep = EdgeProperties(row)
        edge_dict[ep.edge_no] = ep
    end
    return OrderedDict(sort(collect(edge_dict)))
end

# For Matrix input (assume columns are in order: edge_no, susceptance, min_capacity, max_capacity, node1, node2)
"""
    edge_properties_set(data::Matrix)

Constructs a set of edge properties from a matrix input.

# Arguments
- `data::Matrix`: A matrix where each row represents an edge with columns in the order:
  edge_no, susceptance, min_capacity, max_capacity, node1, node2.

# Returns
An OrderedDict where keys are edge numbers and values are EdgeProperties structs.
"""
function edge_properties_set(data::Matrix)
    n_edges = size(data, 1)
    edge_dict = Dict{Int, EdgeProperties}()
    for i in 1:n_edges
        edge_no      = Int(data[i, 1])
        susceptance  = Float64(data[i, 2])
        min_capacity = Float64(data[i, 3])
        max_capacity = Float64(data[i, 4])
        node_tuple   = (Int(data[i, 5]), Int(data[i, 6]))
        ep = EdgeProperties(edge_no, susceptance, min_capacity, max_capacity, node_tuple)
        edge_dict[edge_no] = ep
    end
    ## sort the edge_dict by the edge number
    return OrderedDict(sort(collect(edge_dict)))
end

# Example usage:
# For DataFrame input:
# df = DataFrame(edge_no = [1, 2],
#                susceptance = [0.5, 0.8],
#                min_capacity = [10.0, 15.0],
#                max_capacity = [100.0, 150.0],
#                node1 = [101, 102],
#                node2 = [201, 202])
#
# edges = edge_properties_set(df)
# println(edges)
#
# For Matrix input:
# data = [1 0.5 10.0 100.0 101 201;
#         2 0.8 15.0 150.0 102 202]
#
# edges_matrix = edge_properties_set(data)
# println(edges_matrix)

#############################################################################################################


##############################################################################################################
### function to produce arc incidence matrix
### The function takes the number of nodes and the edge dict as input
### The edge dict key is the edge number and the values are #susceptance, min_capacity, max_capacity, node_tuple
"""
    node_arc_incidence_matrix_generator(n_nodes::Int, edge_properties::OrderedDict)

Generates the arc incidence matrix from a given set of edge properties.

- `n_nodes`: Total number of nodes.
- `edge_properties`: An OrderedDict where each value has a `node_tuple` field.

Each column corresponds to an edge with a -1 at the source node (first element in the tuple) 
and a 1 at the destination node (second element in the tuple).

Returns the transpose of the incidence matrix (i.e. edges as rows) if desired.
"""
function Node_Arc_Incidence_Matrix_Generator(n_nodes::Int, edge_properties::OrderedDict)
    n_edges = length(edge_properties)
    arc_incidence_matrix = zeros(Real, n_nodes, n_edges)
    
    ## assert that the edge number i.e. the key of edge_properties dict is in order and starts from 1 to n_edges continously
    for (i, edge) in enumerate(keys(edge_properties))
        if edge != i
            throw(ArgumentError("Edge numbers must be in order starting from 1 to n_edges."))
        end
    end


    for (edge, properties) in edge_properties
        node1, node2 = properties.node_tuple
        @assert 1 <= node1 <= n_nodes "Node index $node1 out of range (should be between 1 and $n_nodes)"
        @assert 1 <= node2 <= n_nodes "Node index $node2 out of range (should be between 1 and $n_nodes)"
        arc_incidence_matrix[node1, edge] = -1
        arc_incidence_matrix[node2, edge] = 1
    end
    return arc_incidence_matrix'
end

#arc_incidence_matrix = Node_Arc_Incidence_Matrix_Generator(n_nodes, edge_dict)

# # Print the arc incidence matrix to check if it is correct
# println(arc_incidence_matrix)

"""
    susceptance_matrix_generator(n_nodes::Int, edge_properties::OrderedDict)

Generates a susceptance matrix for the network, where off-diagonals are the negative 
susceptance between connected nodes and the diagonal is the sum of susceptances of 
all edges incident to that node.

- `n_nodes`: Total number of nodes.
- `edge_properties`: An OrderedDict with a `susceptance` field and a `node_tuple`.

Returns a symmetric n_nodes × n_nodes matrix.
"""
function Susceptance_Matrix_Generator(n_nodes::Int, edge_properties::OrderedDict)
    susceptance_matrix = zeros(Real, n_nodes, n_nodes)

    ## the order of keys of the edge_properties dict doesn't matter
    for (edge, properties) in edge_properties
        node1 = properties.node_tuple[1]
        node2 = properties.node_tuple[2]
        @assert 1 <= node1 <= n_nodes "Node index $node1 out of range (should be between 1 and $n_nodes)"
        @assert 1 <= node2 <= n_nodes "Node index $node2 out of range (should be between 1 and $n_nodes)"
        
        ### the assumption is that only one line passes between two nodes
        susceptance_matrix[node1, node2] = -properties.susceptance    #### negative sign because the susceptance is negative 
        susceptance_matrix[node2, node1] = -properties.susceptance    

    end

    ### The diagonal elements are the sum of the susceptance of the edges connected to the node

    for i in 1:n_nodes
        susceptance_matrix[i, i] = - sum(susceptance_matrix[i, :])    #### double negative => positive susceptance
    end

    return susceptance_matrix
end

##############################################################################################################

### function to segregate the power generator set according to the bus they are connected to
### input is the power generator set and the edge dict
### The bus to generator dict input has the bus number as the key and the value is a list of power generators connected to that bus
### return dict of dict, where the key is the bus number and the value is a dict of power generators connected to that bus

function Power_Generator_Set_per_Bus(power_generator_set, bus_to_generator_dict)
    power_generator_set_per_bus = Dict()
    for bus in keys(bus_to_generator_dict)
        power_generator_set_per_bus[bus] = Dict()
        for gen in bus_to_generator_dict[bus]
            power_generator_set_per_bus[bus][gen] = power_generator_set[gen]    
        end
    end
    return power_generator_set_per_bus  
end

##############################################################################################################
"""
    bus_total_demand(demand_data)

Create a mapping from bus IDs to their corresponding hourly demand vectors.

# Arguments
- `demand_data`: A matrix or array where each row represents a bus. The first column is the bus ID, and the remaining columns are numeric demand values for each hour (assumed from hour 1 to 24).

# Returns
- An `OrderedDict` where each key is a bus ID and the corresponding value is a `Vector{Float64}` representing the hourly demand information for that bus.

# Notes
- It is assumed that only one demand vector is provided per bus.
- The input data must have at least 2 columns; the first for the bus ID and the rest for hourly demand values.
"""
function bus_total_demand(demand_data)
    n_buses = size(demand_data, 1)
    bus_to_demand_dict = OrderedDict{Any, Any}()
    ### Assume that only one demand vector is given for each bus
    ### the first column is the bus number
    ### the remaining columns are the demand values for each hour starting from 1 to 24
    for i in 1:n_buses
        bus_id = demand_data[i, 1]
        # Store the remaining columns as the demand information.
        bus_to_demand_dict[bus_id] = Vector{Float64}(demand_data[i, 2:end])
    end
    
    return bus_to_demand_dict
end


##############################################################################################################
############################ functions for data generation ###################################################

# Function to generate random vectors within a norm ball
# v is the center of the ball, r is the radius of the ball, n is the number of vectors to generate, p is the norm
# Function to generate random vectors within a norm ball and ensure non-negative values
function generate_vectors_in_norm_ball(v::Vector, r::Real, n::Int; p::Real = 2, seed = 0)
    Random.seed!(seed)
    dim = length(v)
    vectors = Matrix{Float64}(undef, dim, n)
    
    for i in 1:n
        # Generate random direction and random magnitude within the ball
        random_direction = normalize(randn(dim)) # Random unit vector
        random_magnitude = rand()^(1/dim) * r  # Scaled for uniform distribution in ball
        random_vector = random_magnitude * random_direction

        # Generate new vector by adding the random vector within the ball to the original vector
        new_vector = v .+ random_vector
        
        # Truncate any negative values to 0
        new_vector[new_vector .< 0] .= 0.0
        
        # Store the new vector
        vectors[:, i] = new_vector
    end
    
    return vectors
end

function generate_vectors_in_l1_ball(v::Vector{Float64}, fraction::Float64, n::Int; seed=0)
    rng = MersenneTwister(seed)
    dim = length(v)
    total = sum(v)
    radius = fraction * total  # e.g. fraction = 0.1 for 10%

    # Preallocate the output matrix
    vectors = Matrix{Float64}(undef, dim, n)

    for i in 1:n
        # Sample direction from the positive simplex (which is an L1 unit "sphere" in the positive orthant)
        expvals = rand.(rng, Exponential(1.0), dim)
        dir = expvals ./ sum(expvals)  # sum(dir) = 1

        # Choose a random radius for uniform distribution inside L1 ball
        # For an L1 ball, scaling by (rand())^(1/dim) ensures uniform distribution inside the ball
        R = radius * (rand(rng)^(1/dim))

        # Construct the increment vector
        Δ = dir .* R

        # Add the increments to the nominal vector
        new_vector = v .+ Δ

        # Store the result
        vectors[:, i] = new_vector
    end

    return vectors
end

function sort_values_by_keys(dict, key_type = Int)


    sorted_keys = sort(collect(keys(dict))) # Sort the keys
    
    new_dict = OrderedDict{Any, Any}()  
    # Rebuild the OrderedDict with sorted keys and their corresponding values
    for key in sorted_keys
        new_dict[key] = dict[key]
    end
    return new_dict
end

function convert_keys_to_int(dict)

    sorted_keys = sort(parse.(Int, collect(keys(dict)))) # Sort the keys

    new_dict = OrderedDict()
    # Rebuild the OrderedDict with sorted keys and their corresponding values
    for key in sorted_keys
        new_dict[key] = dict["$key"]
    end
    return new_dict
end
##############################################################################################################
"""
    generate_multi_node_scenarios_dict(
        demands::Dict{Int,Vector{<:Real}}, 
        alpha::Real; 
        num_samples::Int=100, 
        seed::Union{Nothing,Int}=nothing
    ) -> Dict{Int,Dict{Int,Vector{Float64}}}

Generate `num_samples` scenarios for multiple buses/nodes. Each bus ID in `demands`
has a nominal 24-hour demand vector (length 24). We perturb these vectors within an
L2-ball of radius `alpha * ||v_b||_2`, where `v_b` is the nominal demand for bus b.
    
# Arguments
- `demands`: A dictionary with bus IDs (Int) as keys, and 24-hour nominal demand 
  vectors (Vector{<:Real}) as values. For example, Vector{Int}, Vector{Float64}, etc.
- `alpha`: A fraction that scales the norm of each bus's 24-hour vector to define
  the radius of its uncertainty set (e.g. 0.1 for 10%).
- `num_samples`: Number of scenarios to generate.
- `seed`: (optional) sets the random seed for reproducibility.

# Returns
A `Dict{Int,Dict{Int,Vector{Float64}}}`, in which:
- The outer dictionary keys are scenario indices (1..num_samples).
- Each value is another dictionary mapping bus ID -> perturbed 24-hour demand vector.
"""
function generate_multi_node_scenarios_dict(
    demands::OrderedDict{}, 
    alpha::Real=0.3; 
    num_samples::Int=100, 
    seed::Union{Nothing,Int}=nothing
)
    # Optionally set the random seed for reproducibility
    if seed !== nothing
        Random.seed!(seed)
    end

    # Prepare the output: scenario_index => (bus_id => demand_vector)
    scenario_dict = Dict{Int,Dict{Int,Vector{Float64}}}()

    # Collect the bus IDs
    bus_ids = collect(keys(demands))

    # For each scenario
    for scenario_idx in 1:num_samples
        scenario_data = Dict{Int,Vector{Float64}}()
        # For each bus, perturb its nominal vector
        for b in bus_ids
            v = demands[b]
            # Convert nominal vector to Float64
            v_float = Vector{Float64}(v)

            # Compute the L2 radius for this bus
            r = alpha * norm(v_float, 2)

            # Random direction in R^24 (Gaussian => normalized)
            z = randn(length(v_float))
            z ./= norm(z, 2)

            # Random magnitude in [0, r]
            ρ = rand() * r

            # Perturbation
            perturbation = ρ .* z

            # Perturbed vector
            scenario_data[b] = max.(0.0, v_float .+ perturbation)
        end

        # Store in outer dictionary
        scenario_dict[scenario_idx] = scenario_data
    end

    return scenario_dict
end

##############################################################################################################

