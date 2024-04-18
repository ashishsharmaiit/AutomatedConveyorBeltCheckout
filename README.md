**Automated Conveyor Belt Checkout Using Computer Vision**



We all have put products on the conveyor belt during the checkout process, where a store employee further scans, weighs and/or count the products to bill you. This process takes unnecessary time and effort from store employee, as computer vision can automatically identify and count the products and a weighing scanner below the belt could weigh the product.

To prototype the idea, I bought a low cost conveyor belt ([link](https://www.vevor.com/belt-conveyor-c_10439/pvc-belt-electric-conveyor-machine-with-stainless-steel-double-guardrail-ce-p_010525771323?utm_source=email&utm_medium=emailnotice&utm_campaign=en_US_orderDelivery_2023-11-14_08-16-50)), and mounted a generic camera on top of it (camera [link](https://www.amazon.com/gp/product/B01DRJXDEA/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1)).

The apparatus looked something like this, with camera mounted on top of the belt.
![IMG_7241](https://github.com/ashishsharmaiit/AutomatedConveyorBeltCheckout/assets/119526028/ff24b45f-b2a1-4d2f-af2f-71480f602fba)

The code was able to identify the products using box detection, image detection, text detection and matching text with a database of products with text, text location and text size.


The code architecture was as below:
1. Identification of bounding boxes for products using a finetuned yolo model
2. Reading the text on the box from Azure AI OCR service.
3. Identifying the product from the text on the product, its relative size in the bounding box.
4. Comparing the text and relative size of the bounding box with a database of products, their text and the sizes.
5. Identifying the products based on above in every frame, and then deduping the products across the video by predicting the product which is moving in the belt vs new products entering.

I was further supplementing the above approach with vector embeddings of products, identifying barcodes and colors. That code is also included in the repository.
