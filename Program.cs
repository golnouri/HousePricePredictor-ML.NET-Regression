using Microsoft.ML;
using Microsoft.ML.Data;
class HouseData
{
    public float Size { get; set; }
    public float Rooms { get; set; }
    public float Price { get; set; }
}
class HousePrediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}
class Program
{
    static void Main()
    {
        var mlContext = new MLContext();
        // Sample Data
        var houseData = new List<HouseData>
        {
            new HouseData { Size = 50, Rooms = 1, Price = 100000000 },
            new HouseData { Size = 60, Rooms = 1, Price = 115000000 },
            new HouseData { Size = 70, Rooms = 2, Price = 140000000 },
            new HouseData { Size = 80, Rooms = 2, Price = 160000000 },
            new HouseData { Size = 90, Rooms = 2, Price = 175000000 },
            new HouseData { Size = 100, Rooms = 3, Price = 210000000 },
            new HouseData { Size = 110, Rooms = 3, Price = 230000000 },
            new HouseData { Size = 120, Rooms = 3, Price = 250000000 },
            new HouseData { Size = 140, Rooms = 4, Price = 290000000 },
            new HouseData { Size = 160, Rooms = 4, Price = 320000000 }
        };
        // Convert to IDataView
        var trainingData = mlContext.Data.LoadFromEnumerable(houseData);
        // Pipeline
        var pipeline = mlContext.Transforms.Concatenate("Features", nameof(HouseData.Size), nameof(HouseData.Rooms)).Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price"));
        // Train
        var model = pipeline.Fit(trainingData);
        // Build prediction engine
        var predictor = mlContext.Model.CreatePredictionEngine<HouseData, HousePrediction>(model);
        Console.WriteLine("House Price Predictor");
        Console.WriteLine("----------------------");
        // Get data from User
        Console.Write("Enter house size (meters): ");
        float size = float.Parse(Console.ReadLine() ?? "0");
        Console.Write("Enter number of rooms: ");
        float rooms = float.Parse(Console.ReadLine() ?? "0");
        var input = new HouseData
        {
            Size = size,
            Rooms = rooms
        };
        var prediction = predictor.Predict(input);
        Console.WriteLine();
        Console.WriteLine($"Estimated price: {prediction.Price:F2}");
    }
}
