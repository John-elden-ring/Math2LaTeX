# Stage 3: Model Architecture (ResNet + Transformer)

## Goal in Plain Words
Design the brain of the system. It needs an "eye" (the image encoder) that looks at the picture of handwritten math and breaks it into visual clues, and a "voice" (the text decoder) that reads those visual clues and writes out the LaTeX equation one token at a time.

---

## How the Architecture Works

### 1. The Image Encoder (`src/model/encoder.py`)
- **Backbone**: We use a standard **ResNet-18**. We remove its final classification layers and keep the convolutional layers.
- **Why ResNet-18?**: It is fast, lightweight, and pre-trained to recognize lines, edges, curves, and textures.
- **Downsampling**: An image of height `128` is downsampled by a factor of 32 to height `4`. A width of `256` becomes width `8`.
- **Channel Projection**: ResNet outputs 512 channels. We use a 1x1 convolution to reduce this to `model_dim = 256` channels so the model stays fast and compact.
- **2D Positional Embeddings**: In math, vertical and horizontal relationships matter (e.g., superscripts are high up, subscripts are low down, denominators are below numerators). We calculate a mathematical 2D sinusoidal position code (row $y$ and column $x$) and add it to each feature before sending it to the decoder.

### 2. The Sequence Decoder (`src/model/decoder.py`)
- **Type**: 4-layer Transformer Decoder with 8 attention heads and hidden dimension 256.
- **Token & Position Embeddings**: Each LaTeX token gets a 256-dimensional vector, plus a 1D position embedding indicating where it appears in the formula.
- **Causal Attention Mask**: Prevents the decoder from cheating by looking ahead at future tokens during training.
- **Cross-Attention**: Allows every generated token to look back at the visual feature grid and focus on the corresponding handwritten symbol.
- **Output Layer**: A linear layer projecting the 256-dimensional vector to logits over the vocabulary.

### 3. Full System (`src/model/img2latex.py`)
- Feeds image to the encoder.
- Uses **Teacher Forcing** during training (giving the decoder the correct previous tokens and asking it to predict the next token).
- Computes Cross-Entropy loss while strictly ignoring `<pad>` tokens.
