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

        static void Main(string[] args)
        {
            try
            {
                //IrisService.PredictIrisFlower();
                //SentimentAnalysis.PredictSentiment();
                TaxiFarePrediction.PredictTaxiFare();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.WriteLine(ex.InnerException != null ? ex.InnerException.Message : "No inner exception");
            }
            Console.ReadKey();
        }

    }
}
