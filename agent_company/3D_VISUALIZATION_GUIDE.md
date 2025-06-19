# 🎮 3D Visualization Guide for H01 Results

## 🚀 **Launching 3D Visualization**

### **Current Status**
- ✅ **3D Viewer**: Launched and running
- ✅ **Comparison Viewer**: Active for segmentation vs proofreading
- 📊 **Available Data**: 5 datasets ready for exploration

---

## 📊 **Available Datasets**

### **Core Segmentation Data**
1. **`segmentation_segmentation.npy`** (64MB)
   - Primary neural network segmentation results
   - Shows initial neuron tracing predictions
   - Color-coded by segment ID

2. **`segmentation_uncertainty.npy`** (64MB)
   - Model confidence scores for each voxel
   - Higher values = lower confidence
   - Useful for identifying problematic regions

### **Proofreading Results**
3. **`proofreading_corrected.npy`** (64MB)
   - Post-proofreading corrected segmentation
   - Improved accuracy after error correction
   - Compare with original segmentation

4. **`proofreading_confidence.npy`** (64MB)
   - Confidence scores after proofreading
   - Shows reliability of corrections
   - Higher values = more reliable corrections

5. **`proofreading_errors.npy`** (16MB)
   - Detected errors in original segmentation
   - Shows regions that needed correction
   - Binary mask of error locations

---

## 🎮 **3D Viewer Controls**

### **Mouse Navigation**
- **Left Click + Drag**: Rotate the 3D view
- **Right Click + Drag**: Pan the view
- **Scroll Wheel**: Zoom in/out
- **Middle Click + Drag**: Zoom (alternative)

### **Keyboard Shortcuts**
- **R**: Reset view to default
- **H**: Toggle help overlay
- **F**: Toggle fullscreen
- **S**: Toggle layer visibility
- **O**: Toggle opacity controls

### **Layer Controls**
- **Visibility Toggle**: Click eye icon to show/hide layers
- **Opacity Slider**: Adjust transparency (0-100%)
- **Color Mapping**: Change color schemes
- **Blending Mode**: Overlay, additive, etc.

---

## 🔍 **Analysis Workflow**

### **1. Initial Exploration**
```bash
# Launch 3D viewer with segmentation
python visualization.py production_output/test_final_advanced --viewer 3d --dataset segmentation_segmentation
```

**What to Look For:**
- Overall structure and connectivity
- Large-scale patterns
- Potential artifacts or errors
- Quality of neuron tracing

### **2. Uncertainty Analysis**
```bash
# View uncertainty overlay
python visualization.py production_output/test_final_advanced --viewer 3d --dataset segmentation_uncertainty
```

**Key Insights:**
- High uncertainty regions need attention
- Low uncertainty = high confidence predictions
- Use for quality assessment

### **3. Proofreading Comparison**
```bash
# Compare original vs corrected
python visualization.py production_output/test_final_advanced --viewer comparison --dataset segmentation_segmentation --dataset2 proofreading_corrected
```

**Analysis Points:**
- What changed during proofreading?
- Error patterns and locations
- Improvement in connectivity

### **4. Error Analysis**
```bash
# Focus on detected errors
python visualization.py production_output/test_final_advanced --viewer 3d --dataset proofreading_errors
```

**Error Types to Identify:**
- False positive connections
- Missing connections
- Boundary errors
- Noise artifacts

---

## 🎯 **Visualization Tips**

### **Optimal Viewing Settings**
1. **Start with Low Opacity** (30-50%)
   - Easier to see internal structure
   - Less overwhelming initially

2. **Use Multiple Layers**
   - Overlay segmentation with uncertainty
   - Compare original vs corrected
   - Toggle between datasets

3. **Focus on Specific Regions**
   - Zoom into interesting areas
   - Rotate to different angles
   - Look for connectivity patterns

### **Color Interpretation**
- **Segmentation**: Each color = different neuron/segment
- **Uncertainty**: Red = high uncertainty, Blue = low uncertainty
- **Errors**: Red = detected errors, Transparent = correct regions
- **Confidence**: Bright = high confidence, Dim = low confidence

---

## 📈 **Quality Assessment**

### **Segmentation Quality Metrics**
1. **Connectivity**: Are neurons properly connected?
2. **Boundaries**: Clear separation between neurons?
3. **Completeness**: No missing segments?
4. **Consistency**: Uniform quality throughout?

### **Proofreading Effectiveness**
1. **Error Reduction**: Fewer errors after correction?
2. **Confidence Improvement**: Higher confidence scores?
3. **Preservation**: Good segments not broken?
4. **Efficiency**: Reasonable number of corrections?

---

## 🔧 **Advanced Features**

### **Slice Views**
- **XY Slice**: Top-down view
- **XZ Slice**: Side view
- **YZ Slice**: Front view
- **3D Rendering**: Full volumetric view

### **Measurement Tools**
- **Distance**: Measure between points
- **Volume**: Calculate segment volumes
- **Surface Area**: Measure boundaries
- **Connectivity**: Count connections

### **Export Options**
- **Screenshots**: Save current view
- **Video**: Record rotation/zoom
- **Data Export**: Save specific regions
- **Reports**: Generate quality metrics

---

## 🚨 **Troubleshooting**

### **Common Issues**
1. **Slow Performance**
   - Reduce data resolution
   - Close other applications
   - Use fewer layers

2. **Memory Issues**
   - Load smaller regions
   - Clear cache
   - Restart viewer

3. **Display Problems**
   - Check graphics drivers
   - Update visualization packages
   - Try different rendering backends

### **Performance Optimization**
```bash
# For better performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python visualization.py --viewer 3d --dataset segmentation_segmentation
```

---

## 📊 **Expected Results**

### **Good Quality Indicators**
- ✅ Smooth neuron boundaries
- ✅ Clear connectivity patterns
- ✅ Low uncertainty in most regions
- ✅ Few proofreading corrections needed

### **Areas for Improvement**
- ⚠️ High uncertainty regions
- ⚠️ Many proofreading corrections
- ⚠️ Broken or incomplete connections
- ⚠️ Boundary artifacts

---

## 🎯 **Next Steps**

### **After 3D Exploration**
1. **Document Findings**: Note interesting regions
2. **Generate Reports**: Use quality analysis tools
3. **Plan Improvements**: Identify model enhancements
4. **Scale Up**: Process larger regions

### **Advanced Analysis**
```bash
# Quality report
python visualization.py production_output/test_final_advanced --viewer quality

# Interactive exploration
python visualization.py production_output/test_final_advanced --viewer interactive

# 2D slice analysis
python visualization.py production_output/test_final_advanced --viewer 2d
```

---

**🎮 Happy Exploring!** 

Your 3D visualization should now be active. Use the controls above to navigate and analyze your H01 segmentation results. The comparison viewer will help you understand the improvements made by the proofreading pipeline. 