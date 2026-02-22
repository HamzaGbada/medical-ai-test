# VLM Evaluation Summary

**Total Samples**: 10  
**CNN Accuracy (on selected)**: 60.00%  
**VLM-GT Agreement**: 50.00%  
**VLM-CNN Agreement**: 40.00%  

## Comparison Table

| Image ID | Ground Truth | CNN Pred | CNN Correct | VLM Pred | VLM=GT | VLM=CNN | Impression (excerpt) |
| -------- | ------------ | -------- | ----------- | -------- | ------ | ------- | -------------------- |
| 131 | Normal | Pneumonia | ❌ | Unclear | ❌ | ❌ | The chest X-ray is difficult to interpret due to poor image quality. There are n... |
| 144 | Normal | Pneumonia | ❌ | Normal | ✅ | ❌ | The chest X-ray shows no acute pulmonary pathology.... |
| 161 | Normal | Pneumonia | ❌ | Pneumonia | ❌ | ✅ | The chest X-ray is of poor quality and limited in its ability to provide a defin... |
| 228 | Pneumonia | Pneumonia | ✅ | Pneumonia | ✅ | ✅ | Based on the limited image quality, it is difficult to provide a definitive diag... |
| 285 | Normal | Normal | ✅ | Pneumonia | ❌ | ❌ | The chest X-ray is of limited quality, making it difficult to provide a definiti... |
| 296 | Pneumonia | Pneumonia | ✅ | Pneumonia | ✅ | ✅ | The chest X-ray is of poor quality and cannot be adequately interpreted. The lun... |
| 356 | Normal | Pneumonia | ❌ | Normal | ✅ | ❌ | The chest X-ray is of poor quality, limiting the ability to provide a definitive... |
| 377 | Normal | Normal | ✅ | Pneumonia | ❌ | ❌ | The chest X-ray is of poor quality, limiting the ability to provide a definitive... |
| 496 | Normal | Normal | ✅ | Pneumonia | ❌ | ❌ | The chest X-ray is limited in quality and cannot provide a definitive diagnosis.... |
| 561 | Pneumonia | Pneumonia | ✅ | Pneumonia | ✅ | ✅ | The chest X-ray is of poor quality and cannot be adequately interpreted. The lun... |
