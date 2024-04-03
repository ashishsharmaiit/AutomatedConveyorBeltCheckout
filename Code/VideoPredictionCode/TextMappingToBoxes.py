class DataCombiner:
    @staticmethod
    def calculate_text_area(bbox):
        x1, y1, x2, y2, x3, y3, x4, y4 = bbox
        area = 0.5 * abs((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))
        return area

    @staticmethod
    def is_inside(text_box, product_box):
        return (min(text_box[0::2]) >= product_box['X1'] and max(text_box[0::2]) <= product_box['X2'] and
                min(text_box[1::2]) >= product_box['Y1'] and max(text_box[1::2]) <= product_box['Y2'])

    @staticmethod
    def calculate_box_area(bbox):
        width = max(bbox[0::2]) - min(bbox[0::2])
        height = max(bbox[1::2]) - min(bbox[1::2])
        return width * height



    def combine_data(self, detections, texts_dict):
        texts_data = texts_dict['text_extraction']
        mapped_data = []
        for text_element in texts_data:
            text_bbox = text_element['bbox']
            text_area = self.calculate_text_area(text_bbox)
            text_length = len(text_element['text'])
            for detection in detections:
                product_bbox = {'X1': detection['X1'], 'Y1': detection['Y1'], 'X2': detection['X2'], 'Y2': detection['Y2']}
                product_area = self.calculate_box_area([product_bbox['X1'], product_bbox['Y1'], product_bbox['X2'], product_bbox['Y2']])
                if self.is_inside(text_bbox, product_bbox):
                    area_ratio = text_area / product_area if product_area != 0 else 0
                    area_ratio_per_character = area_ratio / text_length if text_length != 0 else 0
                    combined_data = {
                        'BoxID': detection['BoxID'],
                        'ObjectID': detection['ObjectID'],
                        'Text': text_element['text'],
                        'Area_Ratio_Per_Character': area_ratio_per_character
                    }
                    mapped_data.append(combined_data)
                    break

        return {'combined_data': mapped_data}
