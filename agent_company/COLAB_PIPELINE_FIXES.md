# Colab Pipeline Fixes and Usage Guide

## Issues Fixed

### 1. Import Path Issues
**Problem**: The original script used `vendor.optimizers` and `vendor.pytorch_connectomics` import paths that don't exist in Colab.

**Solution**: 
- Updated imports to use direct paths: `optimizers.distributed_shampoo` and `pytorch_connectomics.connectomics.config`
- Added proper repository cloning in the setup section
- Added repositories to Python path using `sys.path.extend()`

### 2. Missing Repository Setup
**Problem**: The script didn't clone the required repositories (optimizers, PyTC, SAM 2).

**Solution**:
- Added explicit cloning of all required repositories
- Added proper path management
- Added error handling for missing repositories

### 3. Missing Dependencies
**Problem**: PyTorch Connectomics requires additional dependencies like MONAI, scikit-learn, etc.

**Solution**:
- Added all required PyTC dependencies from setup.py
- Added MONAI >= 0.9.1 for medical imaging support
- Added scientific computing libraries (scipy, scikit-learn, scikit-image)
- Added additional utilities (Cython, yacs, h5py, etc.)

### 4. Configuration File Issues
**Problem**: SAM 2 configuration file path was hardcoded and might not exist.

**Solution**:
- Added dynamic config file discovery
- Added fallback to available config files
- Added better error messages showing available options

### 5. Missing Configuration Parameters
**Problem**: Shampoo optimizer was missing required configuration parameters.

**Solution**:
- Added `PRECONDITION_FREQ` and `START_PRECONDITION_STEP` parameters
- Added default configurations for both H01 and training configs
- Added error handling for missing config files

## Files Created/Updated

### 1. `colab_complete_pipeline.py` (Updated)
- Fixed all import paths
- Added proper repository setup
- Added robust error handling
- Added dynamic config file discovery
- Added default configurations

### 2. `colab_simple_pipeline.py` (New)
- Simplified step-by-step setup script
- Individual import tests
- Better error reporting
- Easier debugging

### 3. `test_imports.py` (New)
- Standalone import test script
- Tests all required imports individually
- Clear success/failure reporting

## Usage Instructions

### Option 1: Use the Simplified Pipeline (Recommended for First Run)
```python
# Run this in Colab to test everything step by step
exec(open('agent_company/colab_simple_pipeline.py').read())
```

### Option 2: Use the Complete Pipeline
```python
# Run this in Colab for the full pipeline
exec(open('agent_company/colab_complete_pipeline.py').read())
```

### Option 3: Test Imports Only
```python
# Run this to test just the imports
exec(open('agent_company/test_imports.py').read())
```

## Expected Output

When running successfully, you should see:

```
======================================================================
COMPLETE CONNECTOMICS PIPELINE SETUP
======================================================================

[1/6] Installing dependencies...
[2/6] Cloning repositories...
[3/6] Setting up SAM 2...
[4/6] Downloading SAM 2 model...
✓ Environment setup complete.

[5/6] Loading libraries...
✓ All libraries imported successfully.

[6/6] Loading configuration...
✓ H01 configuration loaded
✓ Training configuration loaded
✓ Configuration loaded successfully.
```

## Troubleshooting

### If you see "No module named 'vendor'"
- This means the import paths weren't fixed properly
- Make sure you're using the updated `colab_complete_pipeline.py`

### If you see "No module named 'optimizers'"
- The optimizers repository wasn't cloned properly
- Check your internet connection and try again

### If you see "No module named 'sam2'"
- The SAM 2 repository wasn't cloned properly
- Check your internet connection and try again

### If you see "No module named 'pytorch_connectomics'"
- The PyTC repository wasn't cloned properly
- Check your internet connection and try again

### If you see "No module named 'monai'"
- The MONAI dependency wasn't installed properly
- This is required for PyTorch Connectomics
- The script now installs all required dependencies automatically

### If SAM 2 config file is not found
- The script will automatically list available config files
- It will use the first available config file
- If no config files are found, check if SAM 2 was cloned properly

## Next Steps

1. **Test the simplified pipeline first** to ensure all components work
2. **Run the complete pipeline** once the simplified version succeeds
3. **Monitor the output** for any remaining issues
4. **Adjust configurations** as needed for your specific use case

## Key Improvements

1. **Robust Error Handling**: All components now have proper error handling
2. **Dynamic Configuration**: Config files are discovered automatically
3. **Fallback Options**: Default configurations are provided if files are missing
4. **Better Debugging**: Clear error messages and step-by-step testing
5. **Modular Design**: Separate test scripts for easier debugging

The pipeline should now work reliably in Google Colab with proper GPU acceleration and all required dependencies. 