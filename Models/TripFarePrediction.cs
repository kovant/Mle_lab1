using Microsoft.ML.Data;

namespace MLLab1.Models
{
    public class TripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}