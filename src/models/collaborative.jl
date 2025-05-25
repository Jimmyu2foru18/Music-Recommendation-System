using LinearAlgebra
using Statistics
using Random

"""Collaborative Filtering Recommender System implementation."""
mutable struct CollaborativeFilter
    n_factors::Int
    learning_rate::Float64
    regularization::Float64
    user_factors::Matrix{Float64}
    item_factors::Matrix{Float64}
    global_bias::Float64
    user_biases::Vector{Float64}
    item_biases::Vector{Float64}
    
    function CollaborativeFilter(n_users::Int, n_items::Int, n_factors::Int=100;
                                learning_rate::Float64=0.001,
                                regularization::Float64=0.02)
        # Initialize model parameters
        user_factors = randn(n_users, n_factors) .* 0.1
        item_factors = randn(n_items, n_factors) .* 0.1
        global_bias = 0.0
        user_biases = zeros(n_users)
        item_biases = zeros(n_items)
        
        new(n_factors, learning_rate, regularization,
            user_factors, item_factors, global_bias,
            user_biases, item_biases)
    end
end

"""Train the collaborative filtering model."""
function train!(model::CollaborativeFilter, 
                interactions::Matrix{Float64};
                n_epochs::Int=50,
                verbose::Bool=true)
    n_users, n_items = size(interactions)
    
    for epoch in 1:n_epochs
        epoch_loss = 0.0
        n_ratings = 0
        
        # Iterate through all user-item interactions
        for user in 1:n_users, item in 1:n_items
            if interactions[user, item] > 0
                # Compute prediction
                pred = model.global_bias +
                       model.user_biases[user] +
                       model.item_biases[item] +
                       dot(model.user_factors[user, :],
                           model.item_factors[item, :])
                
                # Compute error
                error = interactions[user, item] - pred
                epoch_loss += error^2
                n_ratings += 1
                
                # Update biases
                model.global_bias += model.learning_rate * (error - model.regularization * model.global_bias)
                model.user_biases[user] += model.learning_rate * (error - model.regularization * model.user_biases[user])
                model.item_biases[item] += model.learning_rate * (error - model.regularization * model.item_biases[item])
                
                # Update latent factors
                user_factors = model.user_factors[user, :]
                item_factors = model.item_factors[item, :]
                
                model.user_factors[user, :] += model.learning_rate * 
                    (error * item_factors - model.regularization * user_factors)
                model.item_factors[item, :] += model.learning_rate * 
                    (error * user_factors - model.regularization * item_factors)
            end
        end
        
        if verbose && epoch % 5 == 0
            rmse = sqrt(epoch_loss / n_ratings)
            println("Epoch $epoch: RMSE = $rmse")
        end
    end
end

"""Get recommendations for a specific user."""
function get_recommendations(model::CollaborativeFilter,
                           user_id::Int,
                           n_recommendations::Int=5;
                           exclude_rated::Vector{Int}=Int[])
    predictions = model.global_bias .+
                 model.user_biases[user_id] .+
                 model.item_biases .+
                 model.item_factors * model.user_factors[user_id, :]
    
    # Exclude already rated items
    predictions[exclude_rated] .= -Inf
    
    # Get top N recommendations
    top_n_idx = partialsortperm(predictions, 1:n_recommendations, rev=true)
    
    return top_n_idx, predictions[top_n_idx]
end