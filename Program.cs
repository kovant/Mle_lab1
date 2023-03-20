using System;
using Microsoft.ML;
using MLLab1.Models;

namespace MLLab1
{
    class Program
    {
        static readonly string _trainTripData = "C:\\projects\\MLLab1\\Data\\Train_trip_data.csv";
        static readonly string _testTripData = "C:\\projects\\MLLab1\\Data\\Test_trip_data.csv";

        static void Main(string[] args)
        {
            Console.WriteLine("Prediction of the 'FareAmount' on the basis Fast Tree Regression algorithm.");

            MLContext mlContext = new MLContext(seed: 0);

            var model = Train(mlContext, _trainTripData);

            Evaluate(mlContext, model, _testTripData);

            SinglePrediction(mlContext, model);

            Console.WriteLine(Environment.CurrentDirectory);
        }

        public static ITransformer Train(MLContext mlContext, string trainDataPath)
        {
            Console.WriteLine();
            Console.WriteLine("Start train Model: ");
            Console.WriteLine();

            Console.WriteLine("- Create the Model.");

            // Load train data
            IDataView dataView = mlContext.Data.LoadFromTextFile<Trip>(trainDataPath, hasHeader: true, separatorChar: ',');

            // Create pipeline
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(mlContext.Transforms.Concatenate(
                    "Features",
                    "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
                .Append(mlContext.Regression.Trainers.FastTree());

            Console.WriteLine("- Train the Model ...");

            var model = pipeline.Fit(dataView);

            Console.WriteLine("- End of training.");
            Console.WriteLine();

            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model, string testDataPath)
        {
            // Load test data
            IDataView dataView = mlContext.Data.LoadFromTextFile<Trip>(testDataPath, hasHeader: true, separatorChar: ',');
            // Make prediction
            var predictions = model.Transform(dataView);
            // Make metrics
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine("Model quality metrics evaluation: ");
            Console.WriteLine();
            Console.WriteLine($"- RSquared Score:          {metrics.RSquared:0.##}");
            Console.WriteLine($"- Root Mean Squared Error: {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine();
        }

        private static void SinglePrediction(MLContext mlContext, ITransformer model)
        {
            // Create prediction function and make prediction.
            var predictionFunction = mlContext.Model.CreatePredictionEngine<Trip, TripFarePrediction>(model);

            //Sample:
            //vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
            //CMT,1,1,639,4.3,CRD,14.5
            var tripSample = new Trip
            {
                VendorId = "CMT",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 639,
                TripDistance = 4.3f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual = 15.5
            };

            var prediction = predictionFunction.Predict(tripSample);

            Console.WriteLine("Single prediction: ");
            Console.WriteLine();
            Console.WriteLine($"- Predicted 'Fare amount': {prediction.FareAmount:0.####}, actual value: 15.5");
            Console.WriteLine();
        }
    }
}