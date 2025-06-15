# Image Captioning with PyTorch

A comprehensive implementation of image captioning models using PyTorch, featuring both attention-based models and modern transformer architectures. This project includes multiple approaches to automatically generate descriptive captions for images.

## 🚀 Features

- **Multiple Model Architectures**: Attention-based CNN-LSTM models and fine-tuned BLIP models
- **Complete Training Pipeline**: From data preprocessing to model evaluation
- **Flexible Inference**: Multiple ways to generate captions for new images
- **Modern PyTorch Implementation**: Compatible with PyTorch 2.0+ and modern GPU acceleration
- **Comprehensive Evaluation**: Built-in metrics and submission generation

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for complete dependencies

---

## 📁 Project Structure

```
ImageCaptioning/
├── 📄 Core Scripts
│   ├── train.py                 # Main training script for attention models
│   ├── train_model.py          # Alternative training implementation
│   ├── caption.py              # Generate captions using trained models
│   ├── direct_caption.py       # Simple CNN-LSTM captioning
│   ├── caption_from_checkpoint.py  # Caption using checkpoints
│   └── run_pipeline.py         # Complete training pipeline
│
├── 🧠 Model Components
│   ├── models.py               # Model architectures (Encoder, Decoder, Attention)
│   ├── datasets.py             # Custom dataset classes
│   └── utils.py                # Utility functions and helpers
│
├── 🔧 Data Processing
│   └── create_input_files.py   # Preprocess data for training
│
├── 📊 Analysis & Visualization
│   ├── view_captions.py        # Visualize captions and attention
│   └── image_captioning.ipynb  # Jupyter notebook for exploration
│
├── 📁 Generated Directories
│   ├── data_output/            # Processed data files
│   ├── output_models/          # Saved model checkpoints
│   ├── blip_finetuned_model/   # Fine-tuned BLIP model
│   ├── train/                  # Training images
│   └── test/                   # Test images
│
└── 📋 Configuration
    ├── requirements.txt        # Python dependencies
    ├── .gitignore             # Git ignore rules
    └── README.md              # This file
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ImageCaptioning
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📚 Dataset Setup

1. **Prepare your dataset** - Place training images in `train/train/` and test images in `test/test/`
2. **Create input files**
   ```bash
   python create_input_files.py
   ```
   This will generate:
   - Encoded captions and word mappings
   - Image path mappings
   - Caption length statistics

## 🏃‍♂️ Quick Start

### Option 1: Train from Scratch
```bash
# Preprocess data
python create_input_files.py

# Train the model
python train.py

# Generate captions
python caption.py --encoder_path output_models/encoder.pth --decoder_path output_models/decoder.pth
```

### Option 2: Use Fine-tuned BLIP Model
```bash
# Direct captioning with BLIP
python direct_caption.py --model_path blip_finetuned_model/
```

### Option 3: Complete Pipeline
```bash
# Run the entire pipeline
python run_pipeline.py
```

## 🎯 Model Architectures

### 1. Attention-Based CNN-LSTM Model
- **Encoder**: ResNet-101 backbone for feature extraction
- **Decoder**: LSTM with attention mechanism
- **Attention**: Soft attention to focus on relevant image regions
- **Training**: Custom training loop with gradient clipping and regularization

### 2. BLIP (Bootstrapping Language-Image Pre-training)
- **Modern Architecture**: Transformer-based vision-language model
- **Pre-trained**: Fine-tuned on specific dataset
- **High Performance**: State-of-the-art caption quality

## 📊 Training Configuration

### Hyperparameters
- **Embedding Dimension**: 512
- **Attention Dimension**: 512
- **Decoder Dimension**: 512
- **Batch Size**: 16
- **Learning Rate**: 4e-4 (decoder), 1e-4 (encoder)
- **Epochs**: 3 (default)
- **Dropout**: 0.5

### Training Features
- **Fine-tuning**: Optional encoder fine-tuning
- **Gradient Clipping**: Prevents exploding gradients
- **Attention Regularization**: Doubly stochastic attention
- **Early Stopping**: Based on BLEU-4 score
- **Checkpointing**: Automatic model saving

## 🔍 Inference Methods

### 1. Beam Search Captioning
```python
from caption import caption_images

caption_images(
    encoder_path='output_models/encoder.pth',
    decoder_path='output_models/decoder.pth', 
    word_map_path='data_output/word_map.json',
    test_folder='test/test/',
    output_csv='submission.csv',
    beam_size=3
)
```

### 2. Direct BLIP Captioning
```python
python direct_caption.py --test_folder test/test/ --output_csv blip_submission.csv
```

### 3. Interactive Captioning
Use the Jupyter notebook `image_captioning.ipynb` for interactive exploration and visualization.

## 📈 Evaluation Metrics

The project supports multiple evaluation metrics:
- **BLEU-4**: Standard metric for caption quality
- **METEOR**: Semantic similarity metric
- **CIDEr**: Consensus-based evaluation
- **ROUGE-L**: Longest common subsequence metric

## 🎨 Visualization

Use `view_captions.py` to:
- Visualize attention maps
- Compare model predictions
- Analyze caption quality
- Generate attention heatmaps

## 🚀 Advanced Usage

### Custom Training
```python
# Modify training parameters in train.py
epochs = 5
batch_size = 32
fine_tune_encoder = True
encoder_lr = 1e-5
```

### Model Ensemble
Combine multiple models for better performance:
```python
# Use different models and average predictions
python caption_from_checkpoint.py --ensemble_mode
```

### Data Augmentation
Enhance training with data augmentation:
- Random cropping and resizing
- Color jittering
- Horizontal flipping
- Normalization

## 📁 Generated Files

### Data Output
- `word_map.json`: Vocabulary mapping
- `train_encoded_captions.json`: Preprocessed captions
- `train_image_paths.json`: Image file paths
- `train_caption_lengths.json`: Caption statistics

### Model Output
- `checkpoint_epoch_*.pth`: Training checkpoints
- `encoder.pth` / `decoder.pth`: Trained model weights
- `submission_*.csv`: Prediction files

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in train.py
   batch_size = 8  # or smaller
   ```

2. **Missing Dependencies**
   ```bash
   pip install --upgrade torch torchvision transformers
   ```

3. **Data Loading Issues**
   - Ensure image paths are correct
   - Check file permissions
   - Verify dataset structure

### Performance Optimization

- **GPU Acceleration**: Automatically uses CUDA if available
- **Mixed Precision**: Enable for faster training
- **Data Loading**: Increase `workers` parameter for faster I/O
- **Batch Size**: Adjust based on available GPU memory

## 📊 Results

### Model Performance
- **BLEU-4 Score**: ~0.25-0.30 (attention model)
- **Training Time**: ~2-4 hours on GPU
- **Inference Speed**: ~0.1s per image

### Sample Outputs
Generated captions demonstrate the model's ability to:
- Identify objects and their relationships
- Describe actions and scenes
- Focus attention on relevant image regions
- Generate grammatically correct sentences

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original implementation based on ["Show, Attend and Tell"](https://arxiv.org/abs/1502.03044)
- BLIP model from Salesforce Research
- PyTorch community for excellent documentation
- COCO dataset for training data

## 📧 Contact

For questions or suggestions, please open an issue or contact the maintainers.

---

**Note**: This project requires substantial computational resources for training. Consider using Google Colab or similar cloud platforms if local GPU resources are limited.

Since we're using a pretrained Encoder, we would need to process the images into the form this pretrained Encoder is accustomed to.

Pretrained ImageNet models available as part of PyTorch's `torchvision` module. [This page](https://pytorch.org/docs/master/torchvision/models.html) details the preprocessing or transformation we need to perform – pixel values must be in the range [0,1] and we must then normalize the image by the mean and standard deviation of the ImageNet images' RGB channels.

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```
Also, PyTorch follows the NCHW convention, which means the channels dimension (C) must precede the size dimensions.

We will resize all MSCOCO images to 256x256 for uniformity.

Therefore, **images fed to the model must be a `Float` tensor of dimension `N, 3, 256, 256`**, and must be normalized by the aforesaid mean and standard deviation. `N` is the batch size.

#### Captions

Captions are both the target and the inputs of the Decoder as each word is used to generate the next word.

To generate the first word, however, we need a *zeroth* word, `<start>`.

At the last word, we should predict `<end>` the Decoder must learn to predict the end of a caption. This is necessary because we need to know when to stop decoding during inference.

`<start> a man holds a football <end>`

Since we pass the captions around as fixed size Tensors, we need to pad captions (which are naturally of varying length) to the same length with `<pad>` tokens.

`<start> a man holds a football <end> <pad> <pad> <pad>....`

Furthermore, we create a `word_map` which is an index mapping for each word in the corpus, including the `<start>`,`<end>`, and `<pad>` tokens. PyTorch, like other libraries, needs words encoded as indices to look up embeddings for them or to identify their place in the predicted word scores.

`9876 1 5 120 1 5406 9877 9878 9878 9878....`

Therefore, **captions fed to the model must be an `Int` tensor of dimension `N, L`** where `L` is the padded length.

#### Caption Lengths

Since the captions are padded, we would need to keep track of the lengths of each caption. This is the actual length + 2 (for the `<start>` and `<end>` tokens).

Caption lengths are also important because you can build dynamic graphs with PyTorch. We only process a sequence upto its length and don't waste compute on the `<pad>`s.

Therefore, **caption lengths fed to the model must be an `Int` tensor of dimension `N`**.

### Data pipeline

See `create_input_files()` in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/utils.py).

This reads the data downloaded and saves the following files –

- An **HDF5 file containing images for each split in an `I, 3, 256, 256` tensor**, where `I` is the number of images in the split. Pixel values are still in the range [0, 255], and are stored as unsigned 8-bit `Int`s.
- A **JSON file for each split with a list of `N_c` * `I` encoded captions**, where `N_c` is the number of captions sampled per image. These captions are in the same order as the images in the HDF5 file. Therefore, the `i`th caption will correspond to the `i // N_c`th image.
- A **JSON file for each split with a list of `N_c` * `I` caption lengths**. The `i`th value is the length of the `i`th caption, which corresponds to the `i // N_c`th image.
- A **JSON file which contains the `word_map`**, the word-to-index dictionary.

Before we save these files, we have the option to only use captions that are shorter than a threshold, and to bin less frequent words into an `<unk>` token.

We use HDF5 files for the images because we will read them directly from disk during training / validation. They're simply too large to fit into RAM all at once. But we do load all captions and their lengths into memory.

See `CaptionDataset` in [`datasets.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/datasets.py).

This is a subclass of PyTorch [`Dataset`](https://pytorch.org/docs/master/data.html#torch.utils.data.Dataset). It needs a `__len__` method defined, which returns the size of the dataset, and a `__getitem__` method which returns the `i`th image, caption, and caption length.

We read images from disk, convert pixels to [0,255], and normalize them inside this class.

The `Dataset` will be used by a PyTorch [`DataLoader`](https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) in `train.py` to create and feed batches of data to the model for training or validation.

### Encoder

See `Encoder` in [`models.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py).

We use a pretrained ResNet-101 already available in PyTorch's `torchvision` module. Discard the last two layers (pooling and linear layers), since we only need to encode the image, and not classify it.

We do add an `AdaptiveAvgPool2d()` layer to **resize the encoding to a fixed size**. This makes it possible to feed images of variable size to the Encoder. (We did, however, resize our input images to `256, 256` because we had to store them together as a single tensor.)

Since we may want to fine-tune the Encoder, we add a `fine_tune()` method which enables or disables the calculation of gradients for the Encoder's parameters. We **only fine-tune convolutional blocks 2 through 4 in the ResNet**, because the first convolutional block would have usually learned something very fundamental to image processing, such as detecting lines, edges, curves, etc. We don't mess with the foundations.

### Attention

See `Attention` in [`models.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py).

The Attention network is simple – it's composed of only linear layers and a couple of activations.

Separate linear layers **transform both the encoded image (flattened to `N, 14 * 14, 2048`) and the hidden state (output) from the Decoder to the same dimension**, viz. the Attention size. They are then added and ReLU activated. A third linear layer **transforms this result to a dimension of 1**, whereupon we **apply the softmax to generate the weights** `alpha`.

### Decoder

See `DecoderWithAttention` in [`models.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py).

The output of the Encoder is received here and flattened to dimensions `N, 14 * 14, 2048`. This is just convenient and prevents having to reshape the tensor multiple times.

We **initialize the hidden and cell state of the LSTM** using the encoded image with the `init_hidden_state()` method, which uses two separate linear layers.

At the very outset, we **sort the `N` images and captions by decreasing caption lengths**. This is so that we can process only _valid_ timesteps, i.e., not process the `<pad>`s.

![](./img/sorted.jpg)

We can iterate over each timestep, processing only the colored regions, which are the **_effective_ batch size** `N_t` at that timestep. The sorting allows the top `N_t` at any timestep to align with the outputs from the previous step. At the third timestep, for example, we process only the top 5 images, using the top 5 outputs from the previous step.

This **iteration is performed _manually_ in a `for` loop** with a PyTorch [`LSTMCell`](https://pytorch.org/docs/master/nn.html#torch.nn.LSTM) instead of iterating automatically without a loop with a PyTorch [`LSTM`](https://pytorch.org/docs/master/nn.html#torch.nn.LSTM). This is because we need to execute the Attention mechanism between each decode step. An `LSTMCell` is a single timestep operation, whereas an `LSTM` would iterate over multiple timesteps continously and provide all outputs at once.

We **compute the weights and attention-weighted encoding** at each timestep with the Attention network. In section `4.2.1` of the paper, they recommend **passing the attention-weighted encoding through a filter or gate**. This gate is a sigmoid activated linear transform of the Decoder's previous hidden state. The authors state that this helps the Attention network put more emphasis on the objects in the image.

We **concatenate this filtered attention-weighted encoding with the embedding of the previous word** (`<start>` to begin), and run the `LSTMCell` to **generate the new hidden state (or output)**. A linear layer **transforms this new hidden state into scores for each word in the vocabulary**, which is stored.

We also store the weights returned by the Attention network at each timestep. You will see why soon enough.

# Training

Before you begin, make sure to save the required data files for training, validation, and testing. To do this, run the contents of [`create_input_files.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/create_input_files.py) after pointing it to the the Karpathy JSON file and the image folder containing the extracted `train2014` and `val2014` folders from your [downloaded data](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning#dataset).

See [`train.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/train.py).

The parameters for the model (and training it) are at the beginning of the file, so you can easily check or modify them should you wish to.

To **train your model from scratch**, simply run this file –

`python train.py`

To **resume training at a checkpoint**, point to the corresponding file with the `checkpoint` parameter at the beginning of the code.

Note that we perform validation at the end of every training epoch.

### Loss Function

Since we're generating a sequence of words, we use **[`CrossEntropyLoss`](https://pytorch.org/docs/master/nn.html#torch.nn.CrossEntropyLoss)**. You only need to submit the raw scores from the final layer in the Decoder, and the loss function will perform the softmax and log operations.

The authors of the paper recommend using a second loss – a "**doubly stochastic regularization**". We know the weights sum to 1 at a given timestep. But we also encourage the weights at a single pixel `p` to sum to 1 across _all_ timesteps `T` –

<p align="center">
<img src="./img/doublystochastic.png">
</p>

This means we want the model to attend to every pixel over the course of generating the entire sequence. Therefore, we try to **minimize the difference between 1 and the sum of a pixel's weights across all timesteps**.

**We do not compute losses over the padded regions**. An easy way to do get rid of the pads is to use PyTorch's [`pack_padded_sequence()`](https://pytorch.org/docs/master/nn.html#torch.nn.utils.rnn.pack_padded_sequence), which flattens the tensor by timestep while ignoring the padded regions. You can now aggregate the loss over this flattened tensor.

![](./img/sorted2.jpg)

**Note** – This function is actually used to perform the same dynamic batching (i.e., processing only the effective batch size at each timestep) we performed in our Decoder, when using an `RNN` or `LSTM` in PyTorch. In this case, PyTorch handles the dynamic variable-length graphs internally. You can see an example in [`dynamic_rnn.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling/blob/master/dynamic_rnn.py) in my other tutorial on sequence labeling. We would have used this function along with an `LSTM` in our Decoder if we weren't manually iterating because of the Attention network.

### Early stopping with BLEU

To evaluate the model's performance on the validation set, we will use the automated [BiLingual Evaluation Understudy (BLEU)](http://www.aclweb.org/anthology/P02-1040.pdf) evaluation metric. This evaluates a generated caption against reference caption(s). For each generated caption, we will use all `N_c` captions available for that image as the reference captions.

The authors of the _Show, Attend and Tell_ paper observe that correlation between the loss and the BLEU score breaks down after a point, so they recommend to stop training early on when the BLEU score begins to degrade, even if the loss continues to decrease.

I used the BLEU tool [available in the NLTK module](https://www.nltk.org/_modules/nltk/translate/bleu_score.html).

Note that there is considerable criticism of the BLEU score because it doesn't always correlate well with human judgment. The authors also report the METEOR scores for this reason, but I haven't implemented this metric.

### Remarks

I recommend you train in stages.

I first trained only the Decoder, i.e. without fine-tuning the Encoder, with a batch size of `80`.
I trained for 20 epochs, and the BLEU-4 score peaked at about `23.25` at the 13th epoch. I used the [`Adam()`](https://pytorch.org/docs/master/optim.html#torch.optim.Adam) optimizer with an initial learning rate of `4e-4`.

I continued from the 13th epoch checkpoint allowing fine-tuning of the Encoder with a batch size of `32`. The smaller batch size is because the model is now larger because it contains the Encoder's gradients. With fine-tuning, the score rose to `24.29` in just about 3 epochs. Continuing training would probably have pushed the score slightly higher but I had to commit my GPU elsewhere.

An important distinction to make here is that I'm still supplying the ground-truth as the input at each decode-step during validation, _regardless of the word last generated_. This is called __Teacher Forcing__. While this is commonly used during training to speed-up the process, as we are doing, conditions during validation must mimic real inference conditions as much as possible. I haven't implemented batched inference yet – where each word in the caption is generated from the previously generated word, and terminates upon hitting the `<end>` token.

Since I'm teacher-forcing during validation, the BLEU score measured above on the resulting captions _does not_ reflect real performance. In fact, the BLEU score is a metric designed for comparing naturally generated captions to ground-truth captions of differing length. Once batched inference is implemented, i.e. no Teacher Forcing, early-stopping with the BLEU score will be truly 'proper'.

With this in mind, I used [`eval.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/eval.py) to compute the correct BLEU-4 scores of this model checkpoint on the validation and test sets _without_ Teacher Forcing, at different beam sizes –

Beam Size | Validation BLEU-4 | Test BLEU-4 |
:---: | :---: | :---: |
1 | 29.98 | 30.28 |
3 | 32.95 | 33.06 |
5 | 33.17 | 33.29 |

The test score is higher than the result in the paper, and could be because of how our BLEU calculators are parameterized, the fact that I used a ResNet encoder, and actually fine-tuned the encoder – even if just a little.

Also, remember – when fine-tuning during Transfer Learning, it's always better to use a learning rate considerably smaller than what was originally used to train the borrowed model. This is because the model is already quite optimized, and we don't want to change anything too quickly. I used `Adam()` for the Encoder as well, but with a learning rate of `1e-4`, which is a tenth of the default value for this optimizer.

On a Titan X (Pascal), it took 55 minutes per epoch without fine-tuning, and 2.5 hours with fine-tuning at the stated batch sizes.

### Model Checkpoint

You can download this pretrained model and the corresponding `word_map` [here](https://drive.google.com/open?id=189VY65I_n4RTpQnmLGj7IzVnOF6dmePC).

Note that this checkpoint should be [loaded directly with PyTorch](https://pytorch.org/docs/stable/torch.html?#torch.load), or passed to [`caption.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/caption.py) – see below.

# Inference

See [`caption.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/caption.py).

During inference, we _cannot_ directly use the `forward()` method in the Decoder because it uses Teacher Forcing. Rather, we would actually need to **feed the previously generated word to the LSTM at each timestep**.

`caption_image_beam_search()` reads an image, encodes it, and applies the layers in the Decoder in the correct order, while using the previously generated word as the input to the LSTM at each timestep. It also incorporates Beam Search.

`visualize_att()` can be used to visualize the generated caption along with the weights at each timestep as seen in the examples.

To **caption an image** from the command line, point to the image, model checkpoint, word map (and optionally, the beam size) as follows –

`python caption.py --img='path/to/image.jpeg' --model='path/to/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' --word_map='path/to/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' --beam_size=5`

Alternatively, use the functions in the file as needed.

Also see [`eval.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/eval.py), which implements this process for calculating the BLEU score on the validation set, with or without Beam Search.

### Some more examples

---

![](./img/birds.png)

---

![](./img/salad.png)

---

![](./img/manbike.png)

---

![](./img/catbanana.png)

---

![](./img/firehydrant.png)

---

**The ~~Turing~~ Tommy Test** – you know AI's not really AI because it hasn't watched _The Room_ and doesn't recognize greatness when it sees it.

![](./img/tommy.png)

---

# FAQs

__You said__ ___soft___ __attention. Is there, um, a__ ___hard___ __attention?__

Yes, the _Show, Attend and Tell_ paper uses both variants, and the Decoder with "hard" attention performs marginally better.

In _soft_ attention, which we use here, you're computing the weights `alpha` and using the weighted average of the features across all pixels. This is a deterministic, differentiable operation.

In _hard_ attention, you are choosing to just sample some pixels from a distribution defined by `alpha`. Note that any such probabilistic sampling is non-deterministic or _stochastic_, i.e. a specific input will not always produce the same output. But since gradient descent presupposes that the network is deterministic (and therefore differentiable), the sampling is reworked to remove its stochasticity. My knowledge of this is fairly superficial at this point – I will update this answer when I have a more detailed understanding.

---

__How do I use an attention network for an NLP task like a sequence to sequence model?__

Much like you use a CNN to generate an encoding with features at each pixel, you would use an RNN to generate encoded features at each timestep i.e. word position in the input.

Without attention, you would use the Encoder's output at the last timestep as the encoding for the entire sentence, since it would also contain information from prior timesteps. The Encoder's last output now bears the burden of having to encode the entire sentence meaningfully, which is not easy, especially for longer sentences.

With attention, you would attend over the timesteps in the Encoder's output, generating weights for each timestep/word, and take the weighted average to represent the sentence. In a sequence to sequence task like machine translation, you would attend to the relevant words in the input as you generate each word in the output.

You could also use Attention without a Decoder. For example, if you want to classify text, you can attend to the important words in the input just once to perform the classification.

---

__Can we use Beam Search during training?__

Not with the current loss function, but [yes](https://arxiv.org/abs/1606.02960). This is not common at all.

---

__What is Teacher Forcing?__

Teacher Forcing is when we use the ground truth captions as the input to the Decoder at each timestep, and not the word it generated in the previous timestep. It's common to teacher-force during training since it could mean faster convergence of the model. But it can also learn to depend on being told the correct answer, and exhibit some instability in practice.

It would be ideal to train using Teacher Forcing [only some of the time](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#training-the-model), based on a probability. This is called Scheduled Sampling.

(I plan to add the option).

---

__Can I use pretrained word embeddings (GloVe, CBOW, skipgram, etc.) instead of learning them from scratch?__

Yes, you could, with the `load_pretrained_embeddings()` method in the `Decoder` class. You could also choose to fine-tune (or not) with the `fine_tune_embeddings()` method.

After creating the Decoder in [`train.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/train.py), you should provide the pretrained vectors to `load_pretrained_embeddings()` stacked in the same order as in the `word_map`. For words that you don't have pretrained vectors for, like `<start>`, you can initialize embeddings randomly like we did in `init_weights()`. I recommend fine-tuning to learn more meaningful vectors for these randomly initialized vectors.
  
```python
decoder = DecoderWithAttention(attention_dim=attention_dim,
                               embed_dim=emb_dim,
                               decoder_dim=decoder_dim,
                               vocab_size=len(word_map),
                               dropout=dropout)
decoder.load_pretrained_embeddings(pretrained_embeddings)  # pretrained_embeddings should be of dimensions (len(word_map), emb_dim)
decoder.fine_tune_embeddings(True)  # or False
```
  
Also make sure to change the `emb_dim` parameter from its current value of `512` to the size of your pre-trained embeddings. This should automatically adjust the input size of the decoder LSTM to accomodate them.

---

__How do I keep track of which tensors allow gradients to be computed?__

With the release of PyTorch `0.4`, wrapping tensors as `Variable`s is no longer required. Instead, tensors have the `requires_grad` attribute, which decides whether it is tracked by `autograd`, and therefore whether gradients are computed for it during backpropagation.

- By default, when you create a tensor from scratch, `requires_grad` will be set to `False`.
- When a tensor is created from or modified using another tensor that allows gradients, then `requires_grad` will be set to `True`.
- Tensors which are parameters of `torch.nn` layers will already have `requires_grad` set to `True`.

---

__How do I compute all BLEU (i.e. BLEU-1 to BLEU-4) scores during evaluation?__

You'd need to modify the code in [`eval.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/eval.py) to do this. Please see [this excellent answer](<https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/issues/37#issuecomment-455924998>) by [kmario23](<https://github.com/kmario23>) for a clear and detailed explanation.
