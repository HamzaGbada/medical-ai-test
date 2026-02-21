# VLM Evaluation Summary

**Total Samples**: 10  
**CNN Accuracy (on selected)**: 60.00%  
**VLM-GT Agreement**: 10.00%  
**VLM-CNN Agreement**: 30.00%  

## Comparison Table

| Image ID | Ground Truth | CNN Pred | CNN Correct | VLM Pred | VLM=GT | VLM=CNN | Impression (excerpt) |
| -------- | ------------ | -------- | ----------- | -------- | ------ | ------- | -------------------- |
| 131 | Normal | Pneumonia | ❌ | Pneumonia | ❌ | ✅ | - **Lung Fields:** Clear with no infiltrates or consolidations.
- **Cardiac Silh... |
| 144 | Normal | Pneumonia | ❌ | Unclear | ❌ | ❌ | - **Normal Chest X-ray:** The chest X-ray appears to be normal. There are no sig... |
| 161 | Normal | Pneumonia | ❌ | Pneumonia | ❌ | ✅ | - The chest X-ray appears to show no obvious signs of acute pathology such as pn... |
| 228 | Pneumonia | Pneumonia | ✅ | Normal | ❌ | ❌ | - The chest X-ray shows clear lung fields without infiltrates or consolidations.... |
| 285 | Normal | Normal | ✅ | Pneumonia | ❌ | ❌ | - **Lung Fields:** Clear with no apparent abnormalities.
- **Cardiac Silhouette:... |
| 296 | Pneumonia | Pneumonia | ✅ | Unclear | ❌ | ❌ | - The chest X-ray shows clear lung fields without infiltrates or consolidations.... |
| 356 | Normal | Pneumonia | ❌ | Unclear | ❌ | ❌ | - **Normal Chest X-ray:** The chest X-ray is generally clear with no significant... |
| 377 | Normal | Normal | ✅ | Unclear | ❌ | ❌ | - **Lung Fields:** Clear lung fields with no signs of infiltrates, opacities, or... |
| 496 | Normal | Normal | ✅ | Pneumonia | ❌ | ❌ | - The chest X-ray shows no obvious signs of pneumonia, pulmonary edema, or other... |
| 561 | Pneumonia | Pneumonia | ✅ | Pneumonia | ✅ | ✅ | - **Normal Chest X-ray with Mild Artifacts:** The chest X-ray is generally norma... |
