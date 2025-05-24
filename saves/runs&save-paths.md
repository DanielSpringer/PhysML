base-path = /gpfs/data/fs71925/shepp123/PhysML/saves/

# 02 compare training extent
## 24000 samples, 100 epochs
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS2048_2025-03-08/version_0

## 2000 samples, 1000 epochs
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS2048_2025-03-08/version_1

## 2000 samples, 100 epochs
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS4096_2025-03-08/version_0

## 24000 samples, 1000 epochs
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-08/version_1


# 03 compare latent dimensions
## 128, 64, 32
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-08/version_2

## 128, 64, 32, 16
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-08/version_3

## 128, 64, 32, 8
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-08/version_4

## 128, 32, 16, 4
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-08/version_5


# 04 phase classification
## 04-1 training using all vertices
### 04-1-0 default
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-22/version_1

### 04-1-1 training on uncompressed vertices
--

### 04-1-2 using neural network classifier on uncompressed vertices
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-22/version_1

### 04-1-3 using contrastive autoencoder
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS4096_2025-05-24/version_4

## 04-2 20% test-split within each phase
### 04-2-0 default
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-22/version_2

### 04-2-1 training on uncompressed vertices
--

### 04-2-2 using neural network classifier on compressed vertices
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-22/version_2

## 04-3 exclude SC-phase from autoencoder training
### 04-3-1 exclude SC-phase from classifier training
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-22/version_3

### 04-3-2 include SC-phase in classifier training
#### 04-3-2-0 default
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-22/version_3

#### 04-3-2-1 using neural network classifier
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-22/version_3

#### 04-3-2-2 using contrastive autoencoder (SC)
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS4096_2025-05-15/version_2

#### 04-3-2-3 using contrastive autoencoder (AFM)
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS4096_2025-05-21/version_0

#### 04-3-2-4 using contrastive autoencoder (FM)
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS4096_2025-05-24/version_5


# 05 phase regression
## 05-1 training using all vertices
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-22/version_1

## 05-2 20% test-split within each phase
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-22/version_2

## 05-3 exclude SC-phase from autoencoder training
### 05-3-1 exclude SC-phase from classifier training
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-22/version_3

### 05-3-2 include SC-phase in classifier training
#### 05-3-2-0 default
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-22/version_3

#### 05-3-2-1 using neural network classifier
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS8192_2025-03-22/version_3

#### 05-3-2-2 using contrastive autoencoder (SC)
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS4096_2025-05-15/version_2

#### 05-3-2-3 using contrastive autoencoder (AFM)
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS4096_2025-05-21/version_0

#### 05-3-2-4 using contrastive autoencoder (FM)
vertex_24x6/save_AUTO_ENCODER_VERTEX_24X6_BS4096_2025-05-24/version_5
