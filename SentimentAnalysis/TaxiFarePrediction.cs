using Microsoft.ML.Runtime.Api;
using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Threading.Tasks;

namespace SentimentAnalysis
{
    public class TaxiFarePrediction
    {
        static readonly string _datapath = Path.Combine(Environment.CurrentDirectory, "Datas", "taxi-fare-train.csv");
        static readonly string _testdatapath = Path.Combine(Environment.CurrentDirectory, "Datas", "taxi-fare-test.csv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Datas", "Model.zip");

        static class TestTrips
        {
            internal static readonly TaxiTrip Trip1 = new TaxiTrip
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripDistance = 10.33f,
                PaymentType = "CSH",
                FareAmount = 0 // predict it. actual = 29.5
            };
        }
        public static async Task PredictTaxiFare()
        {
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = await Train();
            Evaluate(model);
            TaxiTripFarePrediction prediction = model.Predict(TestTrips.Trip1);
            Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.FareAmount);
        }
        public static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(_datapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ','));
            pipeline.Add(new ColumnCopier(("FareAmount", "Label")));
            pipeline.Add(new CategoricalOneHotVectorizer("VendorId",
                                             "RateCode",
                                             "PaymentType"));
            pipeline.Add(new ColumnConcatenator("Features",
                                    "VendorId",
                                    "RateCode",
                                    "PassengerCount",
                                    "TripDistance",
                                    "PaymentType"));
            pipeline.Add(new FastTreeRegressor());

            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
            await model.WriteAsync(_modelpath);
            return model;
        }
        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader(_testdatapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine($"Rms = {metrics.Rms}");
        }
    }
    public class TaxiTrip
    {
        [Column("0")]
        public string VendorId;

        [Column("1")]
        public string RateCode;

        [Column("2")]
        public float PassengerCount;

        [Column("3")]
        public float TripTime;

        [Column("4")]
        public float TripDistance;

        [Column("5")]
        public string PaymentType;

        [Column("6")]
        public float FareAmount;
    }

    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}
