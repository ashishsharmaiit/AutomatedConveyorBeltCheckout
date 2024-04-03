import os
import cv2
import json
from pymongo import MongoClient
from DetectingBoxes import BoxIdentifier  
from DetectingText import AzureTextIdentifier 
from TextMappingToBoxes import DataCombiner
from PredictingProducts import ProductNameFinder
from VideoFrameExtractor import VideoFrameExtractor
from TrackingBoxes import ObjectTracker
from CombiningPredictions import ProductPredictionAggregator

# Load configurations
with open('config.json') as config_file:
    config = json.load(config_file)

video_directory = os.path.dirname(config["video_path"])
frame_extractor = VideoFrameExtractor(config["video_path"])
box_identifier = BoxIdentifier(config["model_path"])
text_identifier = AzureTextIdentifier(config["azure_endpoint"], config["azure_key"])
data_combiner = DataCombiner()
client = MongoClient(config["mongo_connection_string"], config["mongo_port"])
object_tracker = ObjectTracker()
prediction_aggregator = ProductPredictionAggregator()


# Main execution
if __name__ == '__main__':

    for frame_count, frame in frame_extractor.extract_frames(frame_interval=5):
        
        detection_results = box_identifier.process_object_detection(frame)
        detections = detection_results.get('detections', [])

        if detections:
            updated_detections = object_tracker.update_objects(detections, frame_count)

            if updated_detections:
                if frame_count % 100 == 0:

                    frame_image_path = os.path.join(video_directory, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_image_path, frame)

                    text_elements = text_identifier.process_text_extraction(frame_image_path, config["test_mode"])

                    combined_results = data_combiner.combine_data(updated_detections, text_elements)

                    product_name_finder = ProductNameFinder(client, config["video_path"])

                    top_words = product_name_finder.get_top_words(combined_results['combined_data'])
                    product_mapping = product_name_finder.find_product_name(top_words)

                    prediction_aggregator.update_predictions(frame_count, product_mapping)
                
                    output = [{'Current_BoxID': k, 'MongoDB_BoxID': v} for k, v in product_mapping.items()]

                    print(json.dumps(output, indent=4))

    final_predictions = prediction_aggregator.get_final_predictions()
    print("Final Predictions:", json.dumps(final_predictions, indent=4))
