using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
namespace SentimentAnalysis
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Datas", "wikipedia-detox-250-line-data.tsv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Datas", "wikipedia-detox-250-line-test.tsv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Datas", "Model.zip");
        static void Main(string[] args)
        {
            try
            {
                //IrisService.PredictIrisFlower();
                Main1(args);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.WriteLine(ex.InnerException != null ? ex.InnerException.Message : "No inner exception");
            }
            Console.ReadKey();
        }
        static async Task Main1(string[] args)
        {
            try
            {
                var model = await Train();
                Evaluate(model);
                Predict(model);
            }
            catch
            {
                throw;
            }
        }
        public static async Task<PredictionModel<SentimentData, SentimentPrediction>> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(_dataPath).CreateFrom<SentimentData>());
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });
            PredictionModel<SentimentData, SentimentPrediction> model =
    pipeline.Train<SentimentData, SentimentPrediction>();
            await model.WriteAsync(_modelpath);
            return model;

        }
        public static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var testData = new TextLoader(_testDataPath).CreateFrom<SentimentData>();
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }
        public static void Predict(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            IEnumerable<SentimentData> sentiments = new[]
                {
                    new SentimentData
                    {
                        SentimentText = "This is the best game I've ever played"
                    },
                    new SentimentData
                    {
                        SentimentText = "He is the best, and the article should say that."

                    },
                    new SentimentData
                    {
                        SentimentText = "It's good to see you."
                    },
                    new SentimentData
                    {
                        SentimentText = "It is very bad what just happened to you."
                    }
                };
            var _prediction = model.Predict(sentiments);
            foreach (var x in _prediction)
            {
                Console.WriteLine("Pred::::1::::" + x.Sentiment);
            }
        }
    }
}
