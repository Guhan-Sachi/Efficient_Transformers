# Vision Transformer Classifier with Streamlit  

## 📌 Overview  
This project provides an **interactive Streamlit application** that allows users to classify images using **efficient Vision Transformer (ViT) models**, specifically **DeiT** and **Swin Transformer**. It also includes **EigenCAM visualization** to highlight the areas the model focuses on.  

## 🚀 Features  
- ✅ Select between **DeiT** and **Swin Transformer** models  
- ✅ **Upload an image** and get the predicted class label  
- ✅ **Displays model performance metrics** (Parameters, FLOPs, Accuracy)  
- ✅ **EigenCAM heatmap visualization** for model explainability  
- ✅ **Optimized for speed and efficiency** with pre-trained ViTs  

## 🎯 Usage  
1. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```
2.    Follow these steps:

        1️⃣ Upload an image via the UI

        2️⃣ Select a Vision Transformer model (DeiT or Swin Transformer)

        3️⃣ View the classification result and EigenCAM visualization

## 📊 Model Comparison  

| Model            | Parameters (M) | FLOPs (G) | Top-1 Accuracy (%) |
|-----------------|--------------|-----------|----------------|
| **DeiT**        | 86M          | 17.5G     | 81.8%          |
| **Swin Transformer** | 88M     | 15.4G     | 83.5%          |

🔍 How It Works

    The uploaded image is preprocessed (resized, normalized).

    It is passed through the selected Vision Transformer for classification.

    The predicted class label is displayed along with inference time.

    EigenCAM visualization highlights which parts of the image the model focused on.

📌 Example Output

    Predicted Class: 🐕 Golden Retriever

    Inference Time: ⏳ 0.45 seconds

📖 Technologies Used

    Deep Learning: PyTorch, Torchvision

    Vision Transformers: Timm (Pre-trained DeiT & Swin Transformer)

    Model Explainability: Grad-CAM, EigenCAM

    UI & Visualization: Streamlit, Matplotlib

🙌 Acknowledgments

    Facebook AI for DeiT

    Microsoft Research for Swin Transformer

    Hugging Face & Timm for pre-trained models

📝 License

This project is open-source under the MIT License.
Feel free to contribute and improve! 🚀
