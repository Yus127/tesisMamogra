# Clasificación de descripciones asociadas a mamografías mediante BioMedCLIP

### Estructura esperada en el conjunto de datos
El conjunto de datos debe almacenarse en una subcarpeta del directorio [.data/](.data/) que contenga las imágenes y los jsons con la información del conjunto de datos en train.json y test.json. Si el conjunto de datos cuenta con un fichero train.json genera el conjunto de entrenamiento y validación a partir de este fichero. Si el conjunto de datos cuenta con un fichero test.json, genera el conjunto de test a partir de este fichero.

📊 Dataset
Data Structure
The dataset should be organized in the .data/ directory with the following structure:

Images: Stored in subdirectories within .data/
Metadata: JSON files containing image paths, labels, and clinical information

JSON Format
train.json / test.json format:

🚀 Key Features


Multi-Class Classification: Distinguishes between normal, benign, and malignant mammographic findings
Transfer Learning: Leverages pre-trained models (VGG, ResNet, DenseNet) adapted for medical imaging
Data Augmentation: Implements medical-aware augmentation techniques to improve model robustness
Comprehensive Evaluation: Includes standard medical AI metrics (sensitivity, specificity, AUC-ROC)

Setup Instructions

Clone the repository:
bashgit clone https://github.com/Yus127/tesisMamogra.git
cd tesisMamogra

Create a virtual environment:
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
bashpip install -r requirements.txt

