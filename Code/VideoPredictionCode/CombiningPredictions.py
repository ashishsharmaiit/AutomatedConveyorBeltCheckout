class ProductPredictionAggregator:
    def __init__(self):
        # Dictionary to store cumulative product predictions for each ObjectID
        self.cumulative_predictions = {}

    def update_predictions(self, frame_count, product_mapping):
        """Update the cumulative predictions with the new product mapping data."""
        for object_id, mongodb_box_id in product_mapping.items():
            if object_id not in self.cumulative_predictions:
                self.cumulative_predictions[object_id] = {}

            if mongodb_box_id not in self.cumulative_predictions[object_id]:
                self.cumulative_predictions[object_id][mongodb_box_id] = 0

            self.cumulative_predictions[object_id][mongodb_box_id] += 1

    def get_final_predictions(self):
        """Calculate the final predictions based on the cumulative data."""
        final_predictions = {}
        for object_id, predictions in self.cumulative_predictions.items():
            # Find the MongoDB BoxID with the highest count for each ObjectID
            best_match = max(predictions, key=predictions.get)
            final_predictions[object_id] = best_match
        return final_predictions

    def reset(self):
        """Reset the cumulative predictions."""
        self.cumulative_predictions.clear()
