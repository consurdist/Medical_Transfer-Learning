# Medical Transfer Learning

Chest X-ray pneumonia classification using transfer learning on a 
decapitated InceptionV3, built as part of Siraj Raval's Machine 
Learning Bootcamp, September 2019.

## Dataset

Kaggle chest X-ray pneumonia dataset — binary classification across 
NORMAL and PNEUMONIA categories, split into train, validation, and 
test sets.

## Approach

InceptionV3 pretrained on ImageNet, imported with `include_top=False` 
to discard the classification head. Two Dense layers (1024 → 512 → 2) 
appended for binary output. Base model layers frozen up to layer 310, 
leaving upper layers trainable. Data pipeline via Keras 
`ImageDataGenerator`. Trained for 10 epochs with Adam optimizer and 
categorical crossentropy loss.

The notebook documents the process honestly including a trainability 
bug that held things up, a shape mismatch on the output layer, and 
a final test accuracy that was, in the immortal words of Meatloaf, 
two out of three ain't bad.

## Status

Completed for bootcamp submission, September 2019. Not maintained.
