using DataFrames, CSV, StatsPlots, MLJ, CategoricalArrays

# prepare data
data = CSV.read(joinpath(@__DIR__, "..", "data", "iris.csv"))
rename!(data, 
 "sepal.length" => :sepal_len ,
 "sepal.width" => :sepal_width,
 "petal.length" => :petal_len,
 "petal.width" => :petal_width,
)
categorical!(data, :variety)
X, y_labels = data[:, Not(:variety)], data[:, :variety]
# train_size = Int(floor(size(data)[1] * 0.75))
train, test = partition(eachindex(y_labels), 0.75, shuffle=true)

# visualizing
gr(size=(600, 600))
@df data corrplot(cols(1:4), gree=false)

# use NearestNeighbors model from MLJ
model = @load KNNClassifier pkg = "NearestNeighbors" verbosity = 1
model.K = 1
knn = machine(model, X, y_labels)
fit!(knn, rows=train)
yhat = predict(knn, X[test,:])
cross_entropy(yhat, y_labels[test]) |> mean