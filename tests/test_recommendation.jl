using DataFrames
using CSV
using Random
using Plots

# Add source directory to load path
pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Import our modules
using data.loader
using data.preprocessor
using models.collaborative
using models.content_based
using visualization.plots
using utils.metrics

# Generate sample data for testing
function generate_sample_data(n_users::Int=100, n_songs::Int=500)
    # Generate song features
    songs = DataFrame(
        song_id = 1:n_songs,
        song_name = ["Song_$i" for i in 1:n_songs],
        duration = rand(120:600, n_songs),
        tempo = rand(60:180, n_songs),
        energy = rand(Float64, n_songs),
        danceability = rand(Float64, n_songs)
    )
    
    # Generate user listening history
    user_history = DataFrame(
        user_id = Int[],
        song_id = Int[],
        play_count = Int[],
        timestamp = Float64[]
    )
    
    # Generate random interactions
    for user_id in 1:n_users
        n_interactions = rand(10:50)
        song_ids = sample(1:n_songs, n_interactions, replace=false)
        
        for song_id in song_ids
            push!(user_history, [
                user_id,
                song_id,
                rand(1:20),  # play count
                rand()  # timestamp (normalized)
            ])
        end
    end
    
    return songs, user_history
end

# Main test function
function test_recommendation_system()
    println("Generating sample data...")
    songs, user_history = generate_sample_data()
    
    # Preprocess data
    println("Preprocessing data...")
    songs_processed = preprocess_songs(songs)
    numerical_features = [:duration, :tempo, :energy, :danceability]
    normalize_features!(songs_processed, numerical_features)
    
    # Create feature matrix for content-based filtering
    feature_matrix = create_feature_matrix(songs_processed, numerical_features)
    
    # Initialize and train content-based recommender
    println("Training content-based recommender...")
    content_recommender = ContentBasedRecommender(numerical_features)
    fit!(content_recommender, songs, feature_matrix)
    
    # Create and train collaborative filtering model
    println("Training collaborative filtering model...")
    interactions, user_to_idx, song_to_idx = create_interaction_matrix(user_history)
    cf_model = CollaborativeFilter(size(interactions)..., 50)
    train!(cf_model, Float64.(interactions), verbose=true)
    
    # Test recommendations for a random user
    test_user_id = rand(1:100)
    println("\nGenerating recommendations for user $test_user_id")
    
    # Get collaborative filtering recommendations
    cf_rec_idx, cf_scores = get_recommendations(cf_model, test_user_id)
    cf_recommendations = DataFrame(
        song_id = cf_rec_idx,
        score = cf_scores,
        song_name = songs.song_name[cf_rec_idx]
    )
    
    println("\nTop 5 Collaborative Filtering Recommendations:")
    println(cf_recommendations[1:5, :])
    
    # Get content-based recommendations
    user_songs = user_history[user_history.user_id .== test_user_id, :]
    cb_rec_idx, cb_scores = get_recommendations(content_recommender, user_songs)
    cb_recommendations = DataFrame(
        song_id = cb_rec_idx,
        score = cb_scores,
        song_name = songs.song_name[cb_rec_idx]
    )
    
    println("\nTop 5 Content-Based Recommendations:")
    println(cb_recommendations[1:5, :])
    
    # Generate visualizations
    println("\nGenerating visualizations...")
    
    # Plot user listening history
    p1 = plot_listening_history(user_songs)
    savefig(p1, joinpath(@__DIR__, "..", "results", "user_history.png"))
    
    # Plot recommendation scores
    p2 = plot_recommendations(cf_recommendations)
    savefig(p2, joinpath(@__DIR__, "..", "results", "cf_recommendations.png"))
    
    # Plot feature importance
    p3 = plot_feature_importance(feature_matrix, string.(numerical_features), vec(sum(interactions, dims=1)))
    savefig(p3, joinpath(@__DIR__, "..", "results", "feature_importance.png"))
    
    println("\nTest completed! Check the 'results' directory for visualization outputs.")
end

# Run the test
test_recommendation_system()