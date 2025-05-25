using DataFrames
using Statistics
using StatsBase

"""Clean and preprocess song data."""
function preprocess_songs(df::DataFrame)
    # Remove duplicates
    unique!(df)
    
    # Handle missing values
    for col in names(df)
        if sum(ismissing.(df[!, col])) > 0
            if eltype(df[!, col]) <: Number
                # Replace missing numerical values with mean
                replace!(df[!, col], missing => mean(skipmissing(df[!, col])))
            else
                # Replace missing categorical values with mode
                replace!(df[!, col], missing => mode(skipmissing(df[!, col])))
            end
        end
    end
    
    return df
end

"""Normalize numerical features to [0,1] range."""
function normalize_features!(df::DataFrame, numerical_cols::Vector{Symbol})
    for col in numerical_cols
        min_val = minimum(df[!, col])
        max_val = maximum(df[!, col])
        if min_val != max_val
            df[!, col] = (df[!, col] .- min_val) ./ (max_val - min_val)
        end
    end
end

"""Create feature matrix from song data."""
function create_feature_matrix(df::DataFrame, feature_cols::Vector{Symbol})
    return Matrix(df[:, feature_cols])
end

"""Split data into training and testing sets."""
function train_test_split(X::Matrix, y::Vector; test_size::Float64=0.2, shuffle::Bool=true)
    n = size(X, 1)
    test_size = floor(Int, test_size * n)
    
    if shuffle
        idx = randperm(n)
        X = X[idx, :]
        y = y[idx]
    end
    
    X_train = X[1:end-test_size, :]
    X_test = X[end-test_size+1:end, :]
    y_train = y[1:end-test_size]
    y_test = y[end-test_size+1:end]
    
    return X_train, X_test, y_train, y_test
end