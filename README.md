Monkeypox-Classification
Four-class skin-lesion image classifier that separates Monkeypox, Measles, Chickenpox and Normal skin, built with transfer learning on a frozen VGG16 (ImageNet weights) in TensorFlow/Keras. A texture-guided DenseNet-121 extension is documented as a proposed (not yet implemented) architecture below.

Dataset
The notebook trains on the dataset Monkeypox Skin Image Dataset (input/monkeypoxskinimagedataset), one folder per class. The image folders in this repo are a 299-image sample of it.

Class	Images in full dataset	Sample in this repo
Normal	293	100
Monkeypox	279	100
Chickenpox	107	58
Measles	91	41
Total	770	299
The classes are imbalanced (Chickenpox and Measles together are about a quarter of the data), so the training set is rebalanced by oversampling. The validation and test sets are left untouched.

Implemented model (Monkeypox.ipynb)
End-to-end pipeline
flowchart TD
    subgraph DATA["Data"]
        A["Kaggle dataset<br/>770 images in 4 class folders"] --> B["get_data: walk folders<br/>DataFrame of file_path and labels"]
    end
    subgraph PREP["Preparation"]
        B --> C["Stratified split 80 / 10 / 10, seed 42<br/>on the original images"]
        C --> D["Train: oversample minority classes<br/>to the largest class count"]
        C --> E["Validation and test: untouched"]
        D --> F["Train generator: VGG16 preprocess_input<br/>+ augmentation, 224 x 224, batch 32"]
        E --> G["Eval generators: VGG16 preprocess_input only<br/>shuffle off"]
    end
    subgraph MODEL["Model"]
        F --> H["VGG16, ImageNet weights<br/>conv base frozen"]
        H --> I["Flatten"]
        I --> J["Dense 4, softmax"]
    end
    subgraph TRAIN["Training and evaluation"]
        J --> K["Adam, categorical cross-entropy<br/>15 epochs"]
        G --> L["Accuracy, confusion matrix,<br/>classification report on test set"]
        K --> L
        L --> M["Save monkeypox.keras"]
    end
Network
flowchart LR
    IN["Input<br/>224 x 224 x 3"] --> V["VGG16 conv base<br/>5 blocks, 13 conv layers<br/>frozen, ImageNet weights"]
    V --> FM["Feature map<br/>7 x 7 x 512"]
    FM --> FL["Flatten<br/>25,088"]
    FL --> OUT["Dense 4<br/>softmax"]
Only the final Dense(4) layer is trained. Parameter counts (computed from the architecture; the notebook also prints model.summary()):

Part	Parameters
VGG16 conv base (frozen)	14,714,688
Dense(4) head (trainable, 25,088 x 4 + 4)	100,356
Total	14,815,044
Methodology
Load. get_data reads one sub-folder per class into a DataFrame of file_path and labels.
Explore. Class counts, a count plot, pie and donut charts, and a 4-images-per-class preview grid.
Split first. Stratified train_test_split (random_state=42): 80% train, then the remaining 20% halved into validation and test. An assertion checks that no file appears in more than one split.
Balance the training set only. Minority classes are sampled with replacement up to the largest training class. Each repeated image gets different random augmentation.
Generators. ImageDataGenerator with VGG16's own preprocess_input. Training adds rotation, flips, shear, zoom and shifts. Class order is fixed with classes=['Chickenpox','Measles','Monkeypox','Normal'] and class_indices is printed.
Model. VGG16 without its top, all layers frozen, then Flatten and Dense(4, softmax).
Train. Adam (default settings), categorical cross-entropy, 15 epochs, validation on the validation generator.
Evaluate. Loss and accuracy on train, validation and test; confusion matrix and classification report (with class names) on the test set.
Export. monkeypox.keras is saved and a download link is shown (the model file is not committed to this repo).
Hyperparameters
Setting	Value
Input size	224 x 224 x 3
Preprocessing	vgg16.preprocess_input
Augmentation	Training only
Batch size	32
Epochs	15
Optimizer / loss	Adam (defaults) / categorical cross-entropy
Base network	VGG16, ImageNet weights, frozen
Head	Flatten, Dense(4, softmax)
Split / seed	80 / 10 / 10 stratified, random_state=42
Results
Leak-free run of the corrected notebook on the 299-image sample in this repo (not the full 770-image Kaggle dataset). Split sizes: 239 train (before balancing), 30 validation, 30 test.

Split	Loss	Accuracy
Train	0.1449	97.50%
Validation (30 images)	6.9146	80.00%
Test (30 images)	0.5739	90.00% (27 of 30)
Test-set classification report:

Class	Precision	Recall	F1	Support
Chickenpox	0.667	1.000	0.800	6
Measles	1.000	0.750	0.857	4
Monkeypox	1.000	0.900	0.947	10
Normal	1.000	0.900	0.947	10
Macro avg	0.917	0.887	0.888	30
How to read this:

Wide uncertainty. With 30 test images, 90% has a 95% Wilson interval of roughly 74% to 96%. One image is 3.3 points. Chickenpox and Measles have only 6 and 4 test images, so their per-class numbers are very noisy.
Overfitting and unstable validation. Training accuracy is 97.5% against 80% validation, and the validation loss (6.9) is very high with validation accuracy bouncing between 77% and 90% across epochs. The 30-image validation set is too small to trust epoch-to-epoch.
Sample only. These results come from the 299-image sample committed here, run locally. The notebook also runs unchanged on the full 770-image Kaggle dataset (about 77 test images), which has not been run, so no full-dataset figure is claimed.
The earlier version of the notebook reported 95.76% test accuracy (118 images). That figure is not reliable: it oversampled before splitting, so the same image could appear in both training and test data. Monkeypox.pdf is an export of that earlier run and is out of date.

What changed from the first version
Leakage fixed. The split now happens before balancing, and only the training set is oversampled.
Correct preprocessing. VGG16's preprocess_input replaces rescale=1/255.
Augmentation is used. The augmenting generator now feeds training (before, it was defined but unused).
Explicit labels. classes=[...] fixes the class order, and class_indices is printed and checked across splits.
Cleanup. Fixed the cell-0 SyntaxError (trailing comma in the import), the mismatched pie/donut labels, the plt.asix typo, an invalid colour code, the %cd kaggle/working export path, and removed the unused custom CNN and unused class weights. Plots now have axis labels and legends.
Remaining limitations
Small test set. 30 images on the sample (about 77 on the full dataset); one extra mistake moves accuracy by 3.3 points (1.3 on the full dataset).
Single split, no cross-validation or confidence interval.
Possible near-duplicates. Different files showing the same lesion or patient could still straddle splits; the dataset has no patient IDs to check this.
Not a diagnostic tool. This is a research/learning project, not validated for clinical use.
Proposed architecture: texture-guided DenseNet-121
Status: proposed design, not implemented. The notebook above trains VGG16 only. This section describes the intended extension; no code or results exist for it yet.

Proposed architecture: DenseNet-121 with wavelet/LBP texture-guided attention

Idea. Skin diseases such as monkeypox, measles and chickenpox differ in lesion texture (pustules, rashes, crusts) as well as colour and shape. A CNN backbone captures general visual features; hand-crafted texture descriptors are used to tell the network where texture matters.

How data flows.

Input image. A 224 x 224 RGB skin image enters two parallel branches.
Texture extraction (left). The image is converted to grayscale, decomposed with a discrete wavelet transform (multi-scale sub-bands), and a local binary pattern (LBP) map is computed. The resulting texture maps are resized to the CNN feature-map size (7 x 7 x T, where T is the number of texture channels).
CNN backbone (right). DenseNet-121 with ImageNet weights, without its classification head, produces a feature map F of 7 x 7 x 1024.
Fusion. The texture maps and F are joined by channel concatenation (the circle marked C). The original diagram used a "+" here; concatenation is the more precise choice because the two inputs have different channel counts. (Element-wise addition would require projecting both to the same number of channels first.)
Attention weights. A 1 x 1 convolution mixes the concatenated channels back to 1024 channels, and a sigmoid squashes the values into (0, 1), giving attention weights A with the same shape as F. (The original diagram labelled this "Conv(IxI)".)
Re-weighting. F is multiplied element-wise by A to give refined features F' = F x A. Locations and channels where texture evidence is strong are emphasised; weak ones are suppressed.
Classifier. Global average pooling reduces F' to a 1,024-vector, and a fully connected layer with softmax outputs probabilities for the four classes.
Corrections to the original sketch.

Original	Corrected
Brace labelled "CNN feature map" around the texture branch, fusion and attention	"Feature extraction and attention fusion" (the brace covers more than the CNN feature map)
"+" fusion node	Channel concatenation, with shapes shown
"Conv(IxI)"	1 x 1 convolution
Attention weights looping back into the multiply node	One-way flow: F and A feed the multiply node, which outputs refined features F'
Unlabelled arrows and boxes	Tensor shapes on every stage
"Fully connected layer", "Output"	Fully connected softmax (4 classes), class probabilities
Photo of a person's hands as the input	Generic input box (no patient image in the diagram)
To make this real, the notebook would need: DWT/LBP texture maps computed per image and resized to 7 x 7; a DenseNet-121 base with densenet.preprocess_input; a custom Keras fusion/attention block; and the same leak-free split and evaluation as above, so the result can be compared fairly with the VGG16 baseline.

Running it
The notebook is written for a Kaggle notebook with a GPU and the dataset attached. To run it elsewhere, set DATA_DIR in the data-loading cell to a folder that contains one sub-folder per class named exactly Chickenpox, Measles, Monkeypox, Normal (this repo's four image folders work as a small sample), and install:

pip install tensorflow pandas numpy scikit-learn seaborn matplotlib pillow
License
MIT, see LICENSE.

