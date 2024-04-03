import os
import re
from pymongo import MongoClient
from operator import itemgetter

class ProductNameFinder:
    def __init__(self, mongo_client, image_path):
        self.db = mongo_client['productDB']
        self.products_collection = self.db['products']
        self.log_file_path = os.path.join(os.path.dirname(image_path), 'process_log.txt')


    def log(self, message):
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(message + "\n")

    def clean_text(self, text):
        """Remove spaces at the beginning, end, and special characters."""
        return re.sub(r'[^a-zA-Z0-9]', '', text).strip()

    def is_relevant_text(self, text):
        """Determine if the text is relevant for searching."""
        cleaned_text = self.clean_text(text)

        # Exclude specific phrases
        excluded_phrases = ['NutritionFacts', 'Calories']
        if any(phrase.lower() in cleaned_text.lower() for phrase in excluded_phrases):
            return False
        
        # Exclude numbers with less than 3 characters
        if cleaned_text.isdigit() and len(cleaned_text) < 3:
            return False

        return True

    def get_top_words(self, combined_data, top_n=6):
        top_words = {}
        for item in combined_data:
            object_id = item['TrackID']
            text = item['Text']
            if not self.is_relevant_text(text):
                self.log(f"Object {object_id}: Excluded word '{text}' from search.")
                continue

            if object_id not in top_words:
                top_words[object_id] = []
            top_words[object_id].append(item)
            self.log(f"TrackID {object_id}: Added word '{text}' with area ratio per character: {item['Area_Ratio_Per_Character']}")

        for object_id, words in top_words.items():
            sorted_words = sorted(words, key=itemgetter('Area_Ratio_Per_Character'), reverse=True)[:top_n]
            top_words[object_id] = sorted_words
            self.log(f"Object {object_id}: Top words - {sorted_words}")

        return top_words


    def find_product_name(self, top_words):
        product_mapping = {}
        for object_id, words in top_words.items():
            product_scores = {}
            matched_texts_per_product = {}  # Track matched texts for each product

            self.log(f"Processing Object {object_id}")

            # Phase 1: Full Match Search
            for word_info in words:
                text = self.clean_text(word_info['Text'])
                text_length_multiplier = len(text)  # Length of the text as a multiplier
                area_ratio_per_character = word_info['Area_Ratio_Per_Character']

                for doc in self.products_collection.find({}):
                    mongodb_box_id = doc['BoxID']
                    if mongodb_box_id not in product_scores:
                        product_scores[mongodb_box_id] = 0
                        matched_texts_per_product[mongodb_box_id] = set()

                    for side in doc['Sides']:
                        for text_entry in side['Texts']:
                            mongodb_text = self.clean_text(text_entry['Text'])
                            if text == mongodb_text and text not in matched_texts_per_product[mongodb_box_id]:
                                score = area_ratio_per_character * text_length_multiplier
                                product_scores[mongodb_box_id] += score
                                matched_texts_per_product[mongodb_box_id].add(text)
                                self.log(f"Full match: '{text}' in Oject {object_id} with MongoDB Box {mongodb_box_id}. Score: {score}")
            
            # Phase 2: Partial Match Search (if no full matches found)
            if not any(product_scores.values()):
                for word_info in words:
                    text = self.clean_text(word_info['Text'])
                    text_length_multiplier = len(text)  # Length of the text as a multiplier

                    for doc in self.products_collection.find({}):
                        mongodb_box_id = doc['BoxID']
                        for side in doc['Sides']:
                            for text_entry in side['Texts']:
                                mongodb_text = self.clean_text(text_entry['Text'])
                                if (text in mongodb_text) and text not in matched_texts_per_product[mongodb_box_id]:
                                    partial_score = (0.5 * word_info['Area_Ratio_Per_Character']) * text_length_multiplier
                                    product_scores[mongodb_box_id] += partial_score
                                    matched_texts_per_product[mongodb_box_id].add(text)
                                    self.log(f"Partial match: '{text}' in Object {object_id} with MongoDB Box {mongodb_box_id}. Score: {partial_score}")

            if not any(product_scores.values()):  # <-- Highlighted modification
                for word_info in words:
                    original_text = word_info['Text']
                    split_words = original_text.split()
                    if len(split_words) > 1:  # Check if the text has multiple words
                        for split_word in split_words:
                            cleaned_split_word = self.clean_text(split_word)
                            split_word_length_multiplier = len(cleaned_split_word)
                            area_ratio_per_character = word_info['Area_Ratio_Per_Character']

                            for doc in self.products_collection.find({}):
                                mongodb_box_id = doc['BoxID']
                                for side in doc['Sides']:
                                    for text_entry in side['Texts']:
                                        mongodb_text = self.clean_text(text_entry['Text'])
                                        if cleaned_split_word in mongodb_text and split_word not in matched_texts_per_product[mongodb_box_id]:
                                            multi_word_partial_score = (0.1 * area_ratio_per_character) * split_word_length_multiplier
                                            product_scores[mongodb_box_id] += multi_word_partial_score
                                            matched_texts_per_product[mongodb_box_id].add(split_word)
                                            self.log(f"Multi-word partial match: '{split_word}' in Object {object_id} with MongoDB Box {mongodb_box_id}. Score: {multi_word_partial_score}")
            if product_scores:
                # Find the highest score
                highest_score = max(product_scores.values())
                # Filter the entries that have the highest score
                top_products = [object_id for object_id, score in product_scores.items() if score == highest_score]

                if len(top_products) == 1:
                    # If only one product has the highest score
                    best_match = top_products[0]
                    product_mapping[object_id] = best_match
                    self.log(f"Object {object_id}: Best match - MongoDB Box {best_match}")
                else:
                    # If more than one product has the highest score or no product found
                    product_mapping[object_id] = 'Can\'t Determine'
                    self.log(f"Box {object_id}: Can't determine the best match due to a tie or no product found.")
            else:
                product_mapping[object_id] = None
                self.log(f"Box {object_id}: No match found")


        return product_mapping
