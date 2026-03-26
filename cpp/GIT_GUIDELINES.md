# Git Repository Guidelines for TrustMark C++

## ? Files to INCLUDE in Git

### Source Code
- `trustmark/*.h` - All header files
- `trustmark/*.cpp` - All implementation files
- `examples/*.cpp` - Example applications

### Build Configuration
- `CMakeLists.txt` - Main build file
- `cmake/*.cmake.in` - CMake configuration templates
- `build.sh` - Build script
- `fetch_ort.sh` - Dependency download script

### Documentation
- `README.md` - Project documentation
- `.gitignore` - Git ignore rules

### Models (Optional - Consider Git LFS)
- `models/*.onnx` - ONNX model files
- `models/.gitkeep` - Keep directory structure

### Directory Structure
- `output/.gitkeep` - Keep output directory structure

---

## ? Files to EXCLUDE (Already in .gitignore)

### Build Artifacts
```
build/                  # Entire build directory
*.o                     # Object files
*.a                     # Static libraries  
*.so, *.dylib, *.dll    # Dynamic libraries
trustmark_example       # Compiled executable
libtrustmark_cpp.*      # Built library
```

### CMake Generated Files
```
CMakeCache.txt
CMakeFiles/
cmake_install.cmake
Makefile
compile_commands.json
TrustMarkCppConfig.cmake
TrustMarkCppConfigVersion.cmake
```

### Dependencies (Fetch via script)
```
onnxruntime/           # Full ONNX Runtime source
third_party/ort/       # Pre-built ONNX Runtime binaries
```

### Runtime Output
```
output/*.jpg           # Generated watermarked images
output/*.png           # Debug outputs
```

### IDE and System Files
```
.cache/
.vscode/
.idea/
.DS_Store
*.swp
```

---

## ?? Current Repository Status

Run this to see what's tracked:
```bash
cd /Users/colmurph/workspaces/github/adobe/trustmark/cpp
git status
```

**Files modified (should be committed):**
- `.gitignore` ?
- `README.md` ?
- `examples/example.cpp` ?
- `trustmark/*.cpp` and `trustmark/*.h` ?

**New files (should be added):**
- `trustmark/execution_provider.h` ?
- `output/.gitkeep` ?

**Files ignored (correct):**
- `build/` ?
- `third_party/` ?
- `onnxruntime/` ?
- `output/*.jpg` and `output/*.png` ?

---

## ?? Quick Commands

### Check what will be committed
```bash
git status
```

### Add source files
```bash
git add trustmark/
git add examples/
git add CMakeLists.txt
git add .gitignore
git add README.md
git add output/.gitkeep
```

### Check if files are ignored
```bash
git check-ignore -v build/
git check-ignore -v output/*.jpg
git check-ignore -v third_party/
```

### Clean untracked files (BE CAREFUL!)
```bash
# Dry run first
git clean -n -d

# Actually remove (if desired)
git clean -f -d
```

---

## ?? Recommended: Git LFS for Models

ONNX models are large binary files. Consider using Git LFS:

```bash
# Install Git LFS
brew install git-lfs  # macOS
# or
sudo apt install git-lfs  # Linux

# Initialize
git lfs install

# Track ONNX models
git lfs track "models/*.onnx"

# Add and commit
git add .gitattributes
git add models/*.onnx
git commit -m "Add ONNX models with Git LFS"
```

---

## ?? CI/CD Setup

For continuous integration:

1. **Don't commit dependencies** - Let CI fetch them
2. **Use `fetch_ort.sh`** in CI pipeline
3. **Cache `third_party/ort/`** between builds
4. **Models**: Either commit with Git LFS or fetch from artifact storage

**Example CI script:**
```bash
#!/bin/bash
# In your CI pipeline

# Fetch dependencies
./fetch_ort.sh

# Build
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Test
./trustmark_example ../images/test.jpg
```

---

## ?? Final Directory Structure

```
cpp/
??? .gitignore                  ? COMMIT
??? README.md                   ? COMMIT
??? CMakeLists.txt              ? COMMIT
??? build.sh                    ? COMMIT
??? fetch_ort.sh                ? COMMIT
??? cmake/                      ? COMMIT
?   ??? *.cmake.in
??? trustmark/                  ? COMMIT (all .h and .cpp)
?   ??? execution_provider.h
?   ??? onnx_session.h/cpp
?   ??? trustmark.h/cpp
?   ??? image_processor.h/cpp
?   ??? bch_ecc.h/cpp
??? examples/                   ? COMMIT
?   ??? example.cpp
??? models/                     ? COMMIT (with Git LFS)
?   ??? .gitkeep
?   ??? *.onnx
??? output/                     ? COMMIT .gitkeep only
?   ??? .gitkeep
??? build/                      ? IGNORE (build artifacts)
??? third_party/                ? IGNORE (dependencies)
??? onnxruntime/                ? IGNORE (source)
```

---

## ? Summary

**Include:** Source code, build config, docs, scripts
**Exclude:** Build artifacts, dependencies, generated outputs, IDE files

The `.gitignore` is now configured correctly! ??
