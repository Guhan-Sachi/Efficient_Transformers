## 📌 Steps to Download and Use the ImageNet Dataset  

### **🔹 Step 1: Download ImageNet Dataset**  
1. Visit the official [ImageNet website](https://image-net.org/download.php).  
2. Register for an account (required for dataset access).  
3. Download the dataset files:  
   - `ILSVRC2012_img_train.tar` (Training images)  
   - `ILSVRC2012_img_val.tar` (Validation images)  
4. Extract the dataset into a local directory.  

### **🔹 Step 2: Load ImageNet in PyTorch**  
1. Use **Torchvision's `ImageFolder` method** to load images.  
2. Apply **image preprocessing** (resize to 224×224, normalize).  
3. Create a **DataLoader** for efficient batch processing.  

### **🔹 Step 3: Use Pre-trained DeiT & Swin Transformer on ImageNet**  
1. Load the **DeiT or Swin Transformer** pre-trained on ImageNet.  
2. Preprocess an image and pass it through the model.  
3. Obtain the **predicted class label** from the model output.  

### **🔹 Step 4: Alternative - Use Smaller ImageNet Subsets**  
If the full ImageNet dataset (~150GB) is too large, use:  
- **[Tiny-ImageNet](https://www.kaggle.com/c/tiny-imagenet)** – A smaller 200-class version.  
- **[ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)** – A compact dataset for quick testing.   
